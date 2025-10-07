import tkinter as tk
from tkinter import messagebox, ttk
from tkinter import simpledialog, Label, Entry, Button
from PIL import Image, ImageTk
import re
import requests
import matplotlib.pyplot as plt
from config import k3s_config_file
from kubernetes import client, config
import openai
import os
import logging
from datetime import datetime, timedelta,timezone
import numpy as np
import skfuzzy as fuzz
import skfuzzy.control as ctrl
import json
import time
import threading
from config import edge_cost_value, cloud_cost_value, cloud_probability_of_failure, edge_probability_of_failure
from config import node_cost_range_random, probability_of_failure, P_MAX, P_IDLE , min_range_reliability,max_range_reliability, locust_master_url
from config import cloud_cpu_core_price, cloud_mem_GB_price, edge_cpu_core_price, edge_mem_GB_price, cloud_nodes_idle, cloud_nodes_max, edge_nodes_idle, edge_nodes_max
import random
import yaml
import shutil
from collections import defaultdict
import math
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
# Cooldown and locking to prevent repeated scaling

node_cost_range = node_cost_range_random  # cost in arbitrary units
node_probability_of_failure = probability_of_failure  # probability of failure
# Set your OpenAI API key
openai.api_key = "XXXXX"
membership_functions = None  # Define a global variable


# Set up logging configuration
logging.basicConfig(
    filename='fuzzy-log',  # Log to the file named "experiment_logs"
    level=logging.INFO,  # Log level: INFO (you can also use DEBUG for more detailed logging)
    format='%(asctime)s - %(levelname)s - %(message)s',  # Log format
    datefmt='%Y-%m-%d %H:%M:%S'  # Date format
)



#======= GET CLUSTER INFO-start
def collect_and_filter_kubernetes_data(print_debug=False):
    try:
        # Specify path to K3s configuration file
        k3s_config_path = k3s_config_file

        # Check if the configuration file exists and is readable
        if not os.path.exists(k3s_config_path) or not os.access(k3s_config_path, os.R_OK):
            raise FileNotFoundError("K3s configuration file does not exist or is not readable.")

        # Load the K3s configuration file
        config.load_kube_config(config_file=k3s_config_path)

        # Create Kubernetes API clients
        core_v1_api = client.CoreV1Api()
        apps_v1_api = client.AppsV1Api()

        # Retrieve list of nodes, pods, and deployments
        node_list = core_v1_api.list_node()
        pod_list = core_v1_api.list_pod_for_all_namespaces(watch=False)
        deployment_list = apps_v1_api.list_deployment_for_all_namespaces(watch=False)

        #-------------------------------------------------
        # Collect node details
        nodes_data = []
        filtered_nodes_data = []
        for node in node_list.items:
            node_info = {
                "name": node.metadata.name,
                "capacity": {
                    "cpu": node.status.capacity.get('cpu', 'Not set'),
                    "memory": node.status.capacity.get('memory', 'Not set')
                },
                "allocatable": {
                    "cpu": node.status.allocatable.get('cpu', 'Not set'),
                    "memory": node.status.allocatable.get('memory', 'Not set')
                },
                "status": node.status.conditions[-1].type if node.status.conditions else "Unknown",
                "roles": node.metadata.labels.get('kubernetes.io/role', 'None'),
                "age": node.metadata.creation_timestamp.strftime(
                    '%Y-%m-%d %H:%M:%S') if node.metadata.creation_timestamp else "Unknown",
                "version": node.status.node_info.kubelet_version if node.status.node_info else "Unknown",
                "internal_ip": next(
                    (addr.address for addr in node.status.addresses if addr.type == "InternalIP"),
                    "Not Available"
                ),
                "external_ip": next(
                    (addr.address for addr in node.status.addresses if addr.type == "ExternalIP"),
                    "Not Available"
                ),
                "os_image": node.status.node_info.os_image if node.status.node_info else "Unknown",
                "kernel_version": node.status.node_info.kernel_version if node.status.node_info else "Unknown",
                "architecture": node.status.node_info.architecture if node.status.node_info else "Unknown"
            }
            nodes_data.append(node_info)

            # Filtered node data for important file
            filtered_nodes_data.append({
                "name": node_info["name"],
                "internal_ip": node_info["internal_ip"],
                "capacity": node_info["capacity"]
            })

        #---------------------------------------------------
        # Collect pod details
        pods_data = []
        important_pods_data = []
        for pod in pod_list.items:
            #==============
            # 🛑 New filter: Only running + ready pods
            if pod.status.phase != "Running":
                continue
            conditions = pod.status.conditions or []
            ready_condition = next((c for c in conditions if c.type == "Ready"), None)
            if not ready_condition or ready_condition.status != "True":
                continue
            if pod.metadata.deletion_timestamp is not None:
                continue

            #==============

            pod_info = {
                "name": pod.metadata.name,
                "namespace": pod.metadata.namespace,  # Keep namespace in original data
                "node": pod.spec.node_name,
                "resources": {
                    "cpu_limit": pod.spec.containers[0].resources.limits.get('cpu', 'Not set')
                    if pod.spec.containers and pod.spec.containers[0].resources and pod.spec.containers[
                        0].resources.limits
                    else "Not set",
                    "memory_limit": pod.spec.containers[0].resources.limits.get('memory', 'Not set')
                    if pod.spec.containers and pod.spec.containers[0].resources and pod.spec.containers[
                        0].resources.limits
                    else "Not set"
                }
            }
            pods_data.append(pod_info)

            # Filter pods with names starting with "microservice"
            if pod.metadata.name.startswith("microservice"):
                pods_data.append(pod_info)
                important_pod_info = {key: value for key, value in pod_info.items() if key != "namespace"}
                important_pods_data.append(important_pod_info)

        #------------------------------------
        # Collect deployment details
        deployments_data = []
        important_deployments_data = []
        for deployment in deployment_list.items:
            deployment_info = {
                "name": deployment.metadata.name,
                "namespace": deployment.metadata.namespace,  # Keep namespace in original data
                "replicas": deployment.spec.replicas
            }
            deployments_data.append(deployment_info)

            # Filter deployments with "microservice" in their name
            if "microservice" in deployment.metadata.name:
                important_deployment_info = {key: value for key, value in deployment_info.items() if
                                             key != "namespace"}
                important_deployments_data.append(important_deployment_info)

        #-----------------------------------------
        # Save original data to JSON
        kubernetes_data = {
            "Kubernetes": {
                "nodes": nodes_data,
                "pods": pods_data,
                "deployments": deployments_data
            }
        }
        with open("k3s_cluster_info_original.json", "w") as file:
            json.dump(kubernetes_data, file, indent=4)
        #print("All details saved to k3s_cluster_info_original.json")
        logging.info("All details saved to k3s_cluster_info_original.json")

        # -----------------------------------------
        if print_debug:
            debug_print_pod_statuses(important_pods_data)
        # -----------------------------------------

        # Save important data to JSON
        important_data = {
            "Kubernetes": {
                "nodes": filtered_nodes_data,
                "pods": important_pods_data,
                "deployments": important_deployments_data
            }
        }
        with open("k3s_cluster_info_important.json", "w") as file:
            json.dump(important_data, file, indent=4)
        print("Important details saved to k3s_cluster_info_important.json")
        logging.info("Important details saved to k3s_cluster_info_important.json")

    # -----------------------------------------



        #-----------------# Attempt to preserve existing cost and reliability
        try:
            with open("k3s_cluster_info_important.json", "r") as existing_file:
                existing_data = json.load(existing_file)
                existing_nodes = {n["name"]: n for n in existing_data.get("Kubernetes", {}).get("nodes", [])}

                for node in filtered_nodes_data:
                    existing = existing_nodes.get(node["name"])
                    if existing and "capacity" in existing:
                        # Preserve these fields if they existed
                        if "node_cost" in existing["capacity"]:
                            node["capacity"]["node_cost"] = existing["capacity"]["node_cost"]
                        if "node_probability_of_failure" in existing["capacity"]:
                            node["capacity"]["node_probability_of_failure"] = existing["capacity"][
                                "node_probability_of_failure"]
        except Exception as e:
            print(f"⚠️ Could not preserve existing node_cost/probability of failure values: {e}")
        #-----------------# Attempt to preserve existing cost and reliability



        # Save filtered node data to JSON
        with open("k3s_important_node_ip.json", "w") as file:
            json.dump(filtered_nodes_data, file, indent=4)
        print("Filtered node IPs saved to k3s_important_node_ip.json")
        logging.info("Filtered node IPs saved to k3s_important_node_ip.json")

    except Exception as e:
        print(f"An error occurred: {e}")
        logging.error(f"An error occurred: {e}")

#======= GET pod placement
def generate_microservice_placement():
    try:
        # File paths
        input_file = "k3s_cluster_info_important.json"
        output_file = "k3s_worker_placement.txt"

        # Read the JSON file
        with open(input_file, "r") as file:
            data = json.load(file)

        # Extract the pods information
        pods = data.get("Kubernetes", {}).get("pods", [])

        # Create a list of worker placements
        worker_sequence = []
        for pod in pods:
            if pod["name"].startswith("microservice"):
                worker_sequence.append(pod['node'])

        # Join the worker placements in the required format
        placement_string = " -> ".join(worker_sequence)

        # Write the output to a text file
        with open(output_file, "w") as file:
            file.write(placement_string)

        print(f"Worker placement saved to {output_file}")

    except Exception as e:
        print(f"An error occurred: {e}")



def get_pod_lifespans():
    # Specify path to K3s configuration file
    k3s_config_path = k3s_config_file  # Replace with your actual config file path

    # Check if the configuration file exists and is readable
    if os.path.exists(k3s_config_path):
        if os.access(k3s_config_path, os.R_OK):
            print("Loading Kubernetes config...")
            config.load_kube_config(config_file=k3s_config_path)

            # Create Kubernetes API clients
            core_v1_api = client.CoreV1Api()

            ###
            try:
                # Fetch the latest list of pods
                print("Fetching pod information...")
                pod_list = core_v1_api.list_pod_for_all_namespaces(watch=False)

                # Read existing data from lifespans.txt
                existing_lifespans = {}
                if os.path.exists("lifespans.txt"):
                    with open("lifespans.txt", "r") as file:
                        for line in file:
                            # Parse existing file lines to extract pod details
                            parts = line.split("- INFO - Pod Name: ")
                            if len(parts) > 1:
                                timestamp = parts[0].strip()
                                pod_info = parts[1].split(", Age (seconds): ")
                                if len(pod_info) == 2:
                                    pod_name = pod_info[0].strip()
                                    existing_lifespans[pod_name] = datetime.strptime(timestamp,
                                                                                     '%Y-%m-%d %H:%M:%S').replace(
                                        tzinfo=timezone.utc)

                # Prepare updated data
                updated_lifespans = {}
                current_time = datetime.now(timezone.utc)  # Ensure this is offset-aware
                for pod in pod_list.items:
                    if pod.metadata.name.startswith("microservice"):
                        pod_name = pod.metadata.name
                        creation_time = pod.metadata.creation_timestamp

                        # Check if pod already exists
                        if pod_name in existing_lifespans:
                            # Use the existing creation timestamp
                            creation_time = existing_lifespans[pod_name]
                        else:
                            # Append new pods with their creation timestamp
                            creation_time = pod.metadata.creation_timestamp

                        # Calculate age dynamically
                        pod_age_seconds = (current_time - creation_time).total_seconds()

                        # Prepare log entry
                        log_entry = {
                            "creation_timestamp": creation_time,
                            "pod_name": pod_name,
                            "age": pod_age_seconds
                        }
                        updated_lifespans[pod_name] = log_entry

                # Write all pods to lifespans.txt
                with open("lifespans.txt", "w") as file:
                    for pod_name, entry in updated_lifespans.items():
                        file.write(
                            f"{entry['creation_timestamp'].strftime('%Y-%m-%d %H:%M:%S')} - INFO - Pod Name: {entry['pod_name']}, Age (seconds): {entry['age']:.6f}\n"
                        )

                #print("Updated lifespans.txt with current pod ages.")
                logging.info("Updated lifespans.txt with current pod ages.")

                #read and print
                #print("\n====================================Contents of updated lifespans.txt=======================================")
                logging.info(
                    "\n====================================Contents of updated lifespans.txt=======================================")
                with open("lifespans.txt", "r") as file:
                    for line in file:
                        #print(line.strip())
                        logging.info("\n====================================Contents of updated lifespans.txt=======================================")
                        logging.info(line.strip())


            except Exception as e:
                print(f"Error fetching or writing pod lifespans: {e}")
                logging.error(f"Error fetching or writing pod lifespans: {e}")
        else:
            print("Insufficient permissions to read the configuration file.")
            logging.error("Insufficient permissions to read the configuration file.")
    else:
        print("K3s configuration file does not exist.")
        logging.error("K3s configuration file does not exist.")

#=========================== Related functions From the main code =========================

def summarize_placement():
    try:
        config.load_kube_config(config_file=k3s_config_file)
        core_v1_api = client.CoreV1Api()
        apps_v1_api = client.AppsV1Api()

        # Load cloud nodes from file
        with open("fixed-files/cloud_nodes.json", "r") as f:
            cloud_nodes = set(json.load(f).get("cloud_nodes", []))

        # Step 1: Get all current split deployment names
        active_deployments = apps_v1_api.list_namespaced_deployment(namespace="default").items
        valid_prefixes = set()

        for dep in active_deployments:
            name = dep.metadata.name
            if name.startswith("microservice") and ("-cloud" in name or "-edge" in name):
                valid_prefixes.add(name)

        # Step 2: Count matching running pods
        pods = core_v1_api.list_namespaced_pod(namespace="default").items
        cloud_count = 0
        edge_count = 0

        for pod in pods:
            name = pod.metadata.name
            node = pod.spec.node_name
            phase = pod.status.phase

            # Match against valid deployments
            for prefix in valid_prefixes:
                if name.startswith(prefix) and phase == "Running" and node:
                    if node in cloud_nodes:
                        cloud_count += 1
                    else:
                        edge_count += 1
                    break  # avoid double counting

        total = cloud_count + edge_count
        '''
        print(f"\n📊 Placement Summary:")
        print(f"🔹 Total microservice pods (split only): {total}")
        if total > 0:
            print(f"☁️  Pods on cloud: {cloud_count} ({cloud_count / total * 100:.1f}%)")
            print(f"🖥️  Pods on edge:  {edge_count} ({edge_count / total * 100:.1f}%)\n")
        else:
            print("⚠️ No microservice pods found to summarize placement.\n")
        '''
    except Exception as e:
        print(f"❌ Error summarizing placement: {e}")
        logging.error(f"❌ Error summarizing placement: {e}")


def label_nodes_to_file(output_file="k3s_cluster_info_important.json"):
    try:
        # Load kubeconfig
        k3s_config_path = k3s_config_file  # or a direct path string if needed
        if os.path.exists(k3s_config_path):
            config.load_kube_config(config_file=k3s_config_path)
        else:
            config.load_kube_config()  # fallback

        v1 = client.CoreV1Api()

        # Load cloud nodes
        with open("fixed-files/cloud_nodes.json", "r") as f:
            cloud_nodes = set(json.load(f).get("cloud_nodes", []))

        with open("fixed-files/fixed_node_attributes.json", "r") as f:
            fixed_node_attrs = json.load(f)

        # Load main cluster info
        with open("k3s_cluster_info_important.json", "r") as f:
            cluster_data = json.load(f)

        nodes = cluster_data["Kubernetes"]["nodes"]
        pods = cluster_data["Kubernetes"]["pods"]

        for node in cluster_data["Kubernetes"]["nodes"]:
            name = node["name"]

            # ------- label as cloud and edge- adding probability of failure to each node
            if name in cloud_nodes:
                node["status"] = "cloud_node"

            else:
                node["status"] = "edge_node"

            # Assign fixed cost and failure probability
            if name in fixed_node_attrs:
                node["capacity"]["node_cost"] = fixed_node_attrs[name]["node_cost"]
                node["capacity"]["node_probability_of_failure"] = fixed_node_attrs[name][
                    "node_probability_of_failure"]
            else:
                # fallback defaults if not in the JSON
                node["capacity"]["node_cost"] = 10
                node["capacity"]["node_probability_of_failure"] = 0.5

        # 2. Compute application-level cost
        node_capacity = {}
        node_usage = {}

        ##
        for node in nodes:
            cpu_str = node["capacity"]["cpu"]
            mem_str = node["capacity"]["memory"]
            cpu = float(cpu_str.replace("m", "")) / 1000 if "m" in cpu_str else float(cpu_str)
            mem = float(mem_str.replace("Ki", "")) / 1024
            node_capacity[node["name"]] = {"cpu": cpu, "memory": mem}

        for pod in pods:
            if not pod["name"].startswith("microservice"):
                continue
            node_name = pod["node"]
            cpu = float(pod["resources"]["cpu_limit"].replace("m", "")) / 1000
            mem = float(pod["resources"]["memory_limit"].replace("Mi", ""))
            if node_name not in node_usage:
                node_usage[node_name] = {"cpu": 0.0, "memory": 0.0}
            node_usage[node_name]["cpu"] += cpu
            node_usage[node_name]["memory"] += mem

        total_cost = 0.0
        monetary_cost = 0.0
        for node in nodes:
            name = node["name"]
            usage = node_usage.get(name, {"cpu": 0.0, "memory": 0.0})

            if node["status"] == "cloud_node":
                cpu_price = cloud_cpu_core_price  # per core per hour
                mem_price = cloud_mem_GB_price   # per MiB per hour
            else:
                cpu_price = edge_cpu_core_price  # edge: 80% cheaper
                mem_price = edge_mem_GB_price  # per MiB per hour

            # Apply cost formula
            cpu_cost = usage["cpu"] * cpu_price
            mem_cost = usage["memory"] * mem_price
            node_consumed_cost = cpu_cost + mem_cost

            # Save per-node consumed cost in dollars
            node["consumed_cost"] = round(node_consumed_cost, 6)

            # Aggregate to application-level cost
            total_cost += node_consumed_cost

            cap = node_capacity.get(name, {"cpu": 1, "memory": 1})
            avg_percent = ((usage["cpu"] / cap["cpu"]) * 100 + (usage["memory"] / cap["memory"]) * 100) / 2

            monetary_cost += (avg_percent / 100.0) * node["capacity"]["node_cost"]

        ##
        # 3. Compute power
        P_idle, P_max = P_IDLE, P_MAX
        node_cpu_usage = {n["name"]: 0.0 for n in nodes}
        pod_cpu_map = {n["name"]: [] for n in nodes}

        for pod in pods:
            if not pod["name"].startswith("microservice"):
                continue
            node_name = pod["node"]
            cpu = float(pod["resources"]["cpu_limit"].replace("m", "")) / 1000
            node_cpu_usage[node_name] += cpu
            pod_cpu_map[node_name].append(cpu)

        total_power = 0.0
        for node in nodes:
            name = node["name"]
            total_cpu = node_capacity[name]["cpu"]
            used_cpu = node_cpu_usage[name]
            cpu_ratio = used_cpu / total_cpu if total_cpu > 0 else 0.0

            # NEW: Override P_idle and P_max per node type
            if node["status"] == "cloud_node":
                P_idle = cloud_nodes_idle
                P_max = cloud_nodes_max
            else:
                P_idle = edge_nodes_idle
                P_max = edge_nodes_max

            node_power = P_idle + (P_max - P_idle) * cpu_ratio
            node["capacity"]["node_power"] = round(node_power, 2)

            consumed_power = sum((c / total_cpu) * node_power for c in pod_cpu_map[name])
            #print(f"this is consumed power: {consumed_power} for pod: {pods}")
            node["consumed_power"] = round(consumed_power, 2)
            total_power += node["consumed_power"]


        # 4. Compute reliability
        node_fail_prob = {n["name"]: n["capacity"]["node_probability_of_failure"] for n in nodes}
        microservice_groups = {}
        for pod in pods:
            match = re.match(r"(microservice\d+)", pod["name"])
            if not match:
                continue
            ms_name = match.group(1)
            microservice_groups.setdefault(ms_name, []).append(pod["node"])

        ##
        app_reliability = 1.0
        reliability_breakdown = {}
        for ms, node_list in microservice_groups.items():
            failure_product = 1.0
            for node in node_list:
                failure_product *= node_fail_prob.get(node, 0)
            ms_reliability = 1 - failure_product
            reliability_breakdown[ms] = {
                "reliability": round(ms_reliability, 6),
                "nodes": node_list
            }
            app_reliability *= ms_reliability

        reliability_breakdown["application_reliability"] = app_reliability

        # 5. Save to output file
        with open(output_file, "w") as f:
            json.dump(cluster_data, f, indent=4)

        # 6. Save metrics
        with open("application_consumed_cost.json", "w") as f:
            json.dump({"application_consumed_cost": round(total_cost, 4)}, f, indent=4)
        with open("application_monetary_cost.json", "w") as f:
            json.dump({"application_monetary_cost": round(monetary_cost, 4)}, f, indent=4)
        with open("application_consumed_power.json", "w") as f:
            json.dump({"application_consumed_power": round(total_power, 4)}, f, indent=4)
        with open("application_reliability.json", "w") as f:
            json.dump(reliability_breakdown, f, indent=4)

        print(f"\n📁 Labeled and enriched cluster saved to: {output_file}")
        print(f"📦 application_consumed_cost = {round(total_cost, 4)}%")
        print(f"💰 application_monetary_cost = ${round(monetary_cost, 4)} per hour")
        print(f"⚡ application_consumed_power = {round(total_power, 4)} W")
        print(f"✅ application_reliability = {round(app_reliability, 4)}")
        print("--------------------------------------------------")

    except Exception as e:
        print(f"❌ Error in label_nodes_to_file: {e}")
        logging.error(f"❌ Error in label_nodes_to_file: {e}")



def save_application_metrics(cluster_info_file, output_json):
    try:
        with open(cluster_info_file, "r") as f:
            data = json.load(f)

        nodes = data["Kubernetes"]["nodes"]

        # Initialize aggregates
        app_consumed_cost = 0.0
        app_monetary_cost = 0.0
        app_consumed_power = 0.0

        for node in nodes:
            app_consumed_cost += node.get("consumed_cost", 0.0)
            app_monetary_cost += (node.get("consumed_cost", 0.0) / 100.0) * node["capacity"].get("node_cost", 0.0)
            app_consumed_power += node.get("consumed_power", 0.0)

        # Read app-level reliability if already saved
        with open("application_reliability.json", "r") as f:
            reliability_data = json.load(f)
            app_reliability = reliability_data.get("application_reliability", 0.0)

        combined_metrics = {
            "application_consumed_cost": round(app_consumed_cost, 4),
            "application_monetary_cost": round(app_monetary_cost, 4),
            "application_consumed_power": round(app_consumed_power, 4),
            "application_reliability": app_reliability
        }

        with open(output_json, "w") as f:
            json.dump(combined_metrics, f, indent=4)

        print(f"📊 Aggregated application metrics saved to {output_json}")
        print(json.dumps(combined_metrics, indent=4))

    except Exception as e:
        print(f"❌ Error in save_application_metrics: {e}")


def update_deployment_replicas(json_input):
    k3s_config_path = k3s_config_file
    if os.path.exists(k3s_config_path) and os.access(k3s_config_path, os.R_OK):
        config.load_kube_config(config_file=k3s_config_path)
        apps_v1_api = client.AppsV1Api()
        try:
            input_data = json.loads(json_input)
            deployment_name = input_data['deployment_name']
            namespace = input_data['namespace']
            replicas = int(input_data['new_replicas'])

            # Fetch and log current replicas
            deployment = apps_v1_api.read_namespaced_deployment(deployment_name, namespace)
            current_replicas = deployment.spec.replicas
            print(f"Current replicas for deployment {deployment_name}: {current_replicas}")
            logging.info(f"Current replicas for deployment {deployment_name}: {current_replicas}")

            # Apply replica update
            body = {
                "spec": {
                    "replicas": replicas
                }
            }
            apps_v1_api.patch_namespaced_deployment(name=deployment_name, namespace=namespace, body=body)
            print(f"✅ Updated {deployment_name} to have {replicas} replicas.")
            logging.info(f"✅ Updated {deployment_name} to have {replicas} replicas.")

        except client.exceptions.ApiException as e:
            print(f"Exception when updating deployment replicas: {e}")
            logging.error(f"Exception when updating deployment replicas: {e}")
        except KeyError as e:
            print(f"Missing key in JSON input: {e}")
            logging.error(f"Missing key in JSON input: {e}")
        except json.JSONDecodeError as e:
            print(f"Error decoding JSON input: {e}")
            logging.error(f"Error decoding JSON input: {e}")
    else:
        print("K3s configuration file does not exist or insufficient permissions to read the configuration file.")
        logging.error("K3s configuration file does not exist or insufficient permissions to read the configuration file.")


def update_deployment_node_selector(json_input, retry_attempts=3):
    k3s_config_path = k3s_config_file
    if os.path.exists(k3s_config_path) and os.access(k3s_config_path, os.R_OK):
        config.load_kube_config(config_file=k3s_config_path)

        api_instance = client.AppsV1Api()
        core_api = client.CoreV1Api()

        for attempt in range(retry_attempts):
            try:
                # Parse JSON input
                input_data = json.loads(json_input)
                deployment_name = input_data['deployment_name']
                deployment_namespace = input_data['deployment_namespace']
                node_selector = input_data['node_selector']

                print(f"Attempting for pod replacement in deployment: {deployment_name} in namespace: {deployment_namespace}")
                logging.info(f"Attempting for pod replacement in deployment: {deployment_name} in namespace: {deployment_namespace}")

                # Read current deployment
                deployment = api_instance.read_namespaced_deployment(name=deployment_name, namespace=deployment_namespace)
                deployment.spec.template.spec.node_selector = node_selector

                # Update deployment with new node selector
                api_instance.replace_namespaced_deployment(name=deployment_name, namespace=deployment_namespace, body=deployment)
                print(f"######### Node selector (for pod replacement) updated for Deployment {deployment_name} in namespace {deployment_namespace}")
                logging.info(f"######### Node selector (for pod replacement) updated for Deployment {deployment_name} in namespace {deployment_namespace}")

                # Delete existing pods to enforce immediate rescheduling
                pods = core_api.list_namespaced_pod(namespace=deployment_namespace)
                for pod in pods.items:
                    if deployment_name in pod.metadata.name:
                        core_api.delete_namespaced_pod(name=pod.metadata.name, namespace=deployment_namespace)
                        print(f"🗑️ Deleted old pod {pod.metadata.name} to enforce placement on new node")
                        logging.info(f"🗑️ Deleted old pod {pod.metadata.name} to enforce placement on new node")

                break  # Exit loop if successful

            except client.exceptions.ApiException as e:
                if e.status == 409 and attempt < retry_attempts - 1:
                    print(f"Conflict error while replacing the pod, retrying... (attempt {attempt + 1})")
                    logging.error(f"Conflict error while replacing the pod, retrying... (attempt {attempt + 1})")
                else:
                    print(f"Exception when updating pod replacement: {e}")
                    logging.error(f"Exception when updating pod replacement: {e}")
                    break
            except KeyError as e:
                print(f"Missing key in JSON input: {e}")
                logging.error(f"Missing key in JSON input: {e}")
                break
            except json.JSONDecodeError as e:
                print(f"Error decoding JSON input: {e}")
                logging.error(f"Error decoding JSON input: {e}")
                break
    else:
        print("K3s configuration file does not exist or insufficient permissions to read the configuration file.")
        logging.error("K3s configuration file does not exist or insufficient permissions to read the configuration file.")

def generate_split_deployments(base_deployment_name, replicas_cloud, replicas_edge):
    return [
        {
            "deployment_name": f"{base_deployment_name}-cloud",
            "namespace": "default",
            "new_replicas": replicas_cloud,
            "node_selector": {"node-group": "cloud"}
        },
        {
            "deployment_name": f"{base_deployment_name}-edge",
            "namespace": "default",
            "new_replicas": replicas_edge,
            "node_selector": {"node-group": "edge"}
        }
    ]

def create_split_deployments(base_deployment_name, replicas_cloud, replicas_edge, sorted_cloud_nodes=None, sorted_edge_nodes=None,namespace="default"):
    config.load_kube_config(config_file=k3s_config_file)
    api = client.AppsV1Api()

    try:
        base_dep = api.read_namespaced_deployment(base_deployment_name, namespace)
        base_spec = base_dep.spec.template
        base_volumes = base_spec.spec.volumes  # 🔁 Copy volumes here


        def create_deployment(name, replicas, node_role):
            label_base = base_deployment_name.replace("-deployment", "")

            ##-----------------------
            # Pick prioritized node (only 1 node is enough for strong preference)
            preferred_node = None
            if node_role == "cloud" and sorted_cloud_nodes:
                preferred_node = sorted_cloud_nodes[0]
            elif node_role == "edge" and sorted_edge_nodes:
                preferred_node = sorted_edge_nodes[0]

            # Build affinity
            affinity = None
            if preferred_node:
                affinity = client.V1Affinity(
                    node_affinity=client.V1NodeAffinity(
                        preferred_during_scheduling_ignored_during_execution=[
                            client.V1PreferredSchedulingTerm(
                                weight=100,
                                preference=client.V1NodeSelectorTerm(
                                    match_expressions=[
                                        client.V1NodeSelectorRequirement(
                                            key="kubernetes.io/hostname",
                                            operator="In",
                                            values=[preferred_node]
                                        )
                                    ]
                                )
                            )
                        ]
                    )
                )


            ##------------------------
            return client.V1Deployment(
                metadata=client.V1ObjectMeta(name=name),
                spec=client.V1DeploymentSpec(
                    replicas=replicas,
                    selector=client.V1LabelSelector(match_labels={"app": label_base}),
                    template=client.V1PodTemplateSpec(
                        metadata=client.V1ObjectMeta(labels={"app": label_base}),
                        spec=client.V1PodSpec(
                            containers=base_spec.spec.containers,
                            volumes=base_volumes,  # ✅ Include original volumes
                            node_selector={"node-group": node_role},
                            affinity=affinity  # ✅ Now actually applying the preferred node!
                        )
                    )
                )
            )

        for name, count, role in [
            (f"{base_deployment_name}-cloud", replicas_cloud, "cloud"),
            (f"{base_deployment_name}-edge", replicas_edge, "edge")
        ]:

            deployment = create_deployment(name, count, role)
            try:
                api.create_namespaced_deployment(namespace=namespace, body=deployment)
                print(f"✅ Created deployment: {name}")

                #-------save deployments into a new files--------------
                print("📂 YAML is being saved to:", os.getcwd())
                # 🔽 Save each to separate file
                dep_dict = api.api_client.sanitize_for_serialization(deployment)
                filename = f"new_{role}_deployments.yml"
                with open(filename, "a") as f:
                    f.write("---\n")  # YAML document separator
                    yaml.dump(dep_dict, f, default_flow_style=False)
                    print(f"🧾 Appended deployment YAML to {filename}")
                # -------save deployments into a new files-------------



            except client.exceptions.ApiException as e:
                if e.status == 409:
                    print(f"ℹ️ Deployment {name} already exists. Skipping creation.")
                else:
                    print(f"❌ Failed to create deployment {name}: {e}")
    except Exception as e:
        print(f"❌ Error creating split deployments for {base_deployment_name}: {e}")


def take_action():
    try:
        logging.info("🟢 take_action function started...")
        print("🟢 take_action function started...")

        # ✅ Step 0: Label all nodes before doing anything
        #label_nodes_adds_on()

        # ✅ Step 1: Load fuzzy output
        try:
            with open("fuzzy_output.json", "r") as file:
                fuzzy_data = json.load(file)
            scale_ratio = float(fuzzy_data.get("new_replicas", 1))
            placement_ratio = float(fuzzy_data.get("new_placement", 0.5))
        except Exception as e:
            logging.error(f"❌ Error reading fuzzy_output.json: {e}")
            return

        # ✅ Step 2: Load cluster info
        with open("before_cluster_snapshot.json", "r") as f:
            cluster_data = json.load(f)

        # ✅ NEW: Prioritize nodes for smarter placement
        sorted_cloud_nodes, sorted_edge_nodes = prioritize_nodes_for_placement(cluster_data)
        print(f"🧠 Preferred cloud nodes (by cost): {sorted_cloud_nodes}")
        print(f"🧠 Preferred edge nodes (by reliability): {sorted_edge_nodes}")

        # Find all microservice deployments (base names)
        # ✅ Unified handling of deployments: supports both original and split forms
        base_deployments_set = set()
        for d in cluster_data["Kubernetes"]["deployments"]:
            name = d["name"]
            if name.startswith("microservice"):
                # Normalize base name
                if name.endswith("-cloud"):
                    base_name = name.rsplit("-cloud", 1)[0]
                elif name.endswith("-edge"):
                    base_name = name.rsplit("-edge", 1)[0]
                else:
                    base_name = name
                base_deployments_set.add(base_name)

        base_deployments = sorted(base_deployments_set)
        print(f"\n🔍 Detected base deployments: {base_deployments}\n")
        logging.info(f"\n🔍 Detected base deployments: {base_deployments}\n")

        for base_name in base_deployments:
            # Find current replica count from cluster info
            # Look for original or split variants
            original = next((d for d in cluster_data["Kubernetes"]["deployments"] if d["name"] == base_name), None)
            cloud = next((d for d in cluster_data["Kubernetes"]["deployments"] if d["name"] == f"{base_name}-cloud"),
                         None)
            edge = next((d for d in cluster_data["Kubernetes"]["deployments"] if d["name"] == f"{base_name}-edge"),
                        None)

            # Determine current replica count
            if original:
                current_replicas = original.get("replicas", 1)
            elif cloud and edge:
                current_replicas = cloud.get("replicas", 0) + edge.get("replicas", 0)
            else:
                print(f"❌ No valid deployment found for {base_name}, skipping...")
                logging.warning(f"❌ No valid deployment found for {base_name}, skipping...")
                continue

            print(f"➡️ Working on {base_name}: current replicas = {current_replicas}")
            logging.info(f"➡️ Working on {base_name}: current replicas = {current_replicas}")

            #------------------------------------------------
            replica_bounds = load_replica_bounds()
            print(f"This is replica_bounds: {replica_bounds}")
            current_users = get_current_user_count()
            print(f"This is current_users: {current_users}")

            # Strip everything after the microservice number (e.g., "microservice1-deployment" → "microservice1")

            base_name_cleaned = re.match(r"(microservice\d+)", base_name).group(1)
            print(f"🧩 Cleaned base_name: {base_name_cleaned}")
            print(f"📂 Available services in replica_bounds: {list(replica_bounds.keys())}")

            valid_levels = list(replica_bounds[base_name_cleaned].keys())
            print(f"This is valid_levels: {valid_levels}")
            closest_level = get_closest_user_level(current_users, valid_levels)
            print(f"This is closest_level :{closest_level }")

            min_r = replica_bounds[base_name_cleaned][closest_level]["min"]
            print(f"This is min_r : {min_r }")
            max_r = replica_bounds[base_name_cleaned][closest_level]["max"]
            print(f"This is max_r: {max_r}")

            scaled_replicas = round(min_r + scale_ratio * (max_r - min_r))
            print(f"Scaled replicas for {base_name_cleaned}: {scaled_replicas}")
            # ------------------------------------------------

            # ------------------------------------------------



            cloud_replicas = round(scaled_replicas * placement_ratio)

            edge_replicas = scaled_replicas - cloud_replicas



            #-----------------print------------------------
            print(f"This is the current_replicas in take_action function: {current_replicas}")
            print(f"This is the scaled_replicas in take_action function: {scaled_replicas}")
            print(f"This is the cloud_replicas in take_action function: {cloud_replicas}")
            print(f"This is the edge_replicas in take_action function: {edge_replicas}")

            print(f"👥 Detected {current_users} users → closest defined level: {closest_level}")
            print(f"📌 {base_name} bounds → min: {min_r}, max: {max_r}")
            #----------------------------------------------


            sorted_cloud_nodes, sorted_edge_nodes = prioritize_nodes_for_placement(cluster_data)
            # ✅ Step 3: Create split deployments
            create_split_deployments(base_name, cloud_replicas, edge_replicas, sorted_cloud_nodes, sorted_edge_nodes)  # ✅ NEW

            # ✅ Step 4: Update replicas and node selectors for split deployments
            split_plan = generate_split_deployments(base_name, cloud_replicas, edge_replicas)

            for entry in split_plan:
                update_deployment_replicas(json.dumps({
                    "deployment_name": entry["deployment_name"],
                    "namespace": entry["namespace"],
                    "new_replicas": entry["new_replicas"]
                }))
                update_deployment_node_selector(json.dumps({
                    "deployment_name": entry["deployment_name"],
                    "deployment_namespace": entry["namespace"],
                    "node_selector": entry["node_selector"]
                }))

            # ✅ Step 5: Wait for stability (optional but safer)
            time.sleep(1)

            # ✅ Step 6: Delete original deployment
            try:
                apps_api = client.AppsV1Api()
                apps_api.delete_namespaced_deployment(name=base_name, namespace="default")
                print(f"🗑️ Deleted original deployment {base_name}")
                logging.info(f"🗑️ Deleted original deployment {base_name}")
            except Exception as e:
                print(f"❌ Failed to delete original deployment {base_name}: {e}")
                logging.error(f"❌ Failed to delete original deployment {base_name}: {e}")
        summarize_placement()
        logging.info("✅ All microservices updated with replica scaling and placement.")
    except Exception as e:
        logging.error(f"❌ Exception in take_action: {e}")

#----------------------------------------------
#for loading the locust file
def load_replica_bounds(file_path="fixed-files/replica_bounds.json"):
    with open(file_path, "r") as f:
        return json.load(f)

def get_closest_user_level(current_users, valid_levels):
    return min(valid_levels, key=lambda x: abs(current_users - int(x)))

def get_current_user_count(locust_master_url=f"http://{locust_master_url}:8089"):
    try:
        response = requests.get(f"{locust_master_url}/stats/requests")
        response.raise_for_status()
        data = response.json()
        return data.get("user_count", 0)
    except Exception as e:
        print(f"❌ Could not fetch Locust traffic info: {e}")
        return 0
#-----------------------------------------------------

def prioritize_nodes_for_placement(cluster_data):
    """
    Returns sorted lists of cloud and edge nodes based on:
    - Lowest cost for cloud
    - Lowest failure probability for edge
    """

    cloud_nodes = []
    edge_nodes = []

    for node in cluster_data["Kubernetes"]["nodes"]:
        node_name = node["name"]
        status = node.get("status", "")
        cost = node["capacity"].get("node_cost", float("inf"))
        failure_prob = node["capacity"].get("node_probability_of_failure", 1.0)

        if status == "cloud_node":
            cloud_nodes.append((node_name, cost))
        elif status == "edge_node":
            edge_nodes.append((node_name, failure_prob))

    # Sort cloud by lowest cost, edge by lowest failure probability
    sorted_cloud = [n for n, _ in sorted(cloud_nodes, key=lambda x: x[1])]
    sorted_edge = [n for n, _ in sorted(edge_nodes, key=lambda x: x[1])]

    return sorted_cloud, sorted_edge


def print_node_labels():
    try:
        k3s_config_path = k3s_config_file
        config.load_kube_config(config_file=k3s_config_path)
        core_v1_api = client.CoreV1Api()

        # Load cloud nodes from fixed file
        with open("fixed-files/cloud_nodes.json", "r") as f:
            cloud_nodes = set(json.load(f)["cloud_nodes"])

        logging.info("\n===== NODE LABELS IN KUBERNETES =====")
        for node in core_v1_api.list_node().items:
            node_name = node.metadata.name
            labels = node.metadata.labels or {}

            # Determine correct node-group value
            new_label_value = "cloud" if node_name in cloud_nodes else "edge"

            # Apply the label live if it's not already correct
            current_label = labels.get("node-group")
            if current_label != new_label_value:
                patch_body = {
                    "metadata": {
                        "labels": {
                            "node-group": new_label_value
                        }
                    }
                }
                core_v1_api.patch_node(node_name, patch_body)
                print(f"🏷️ Updated node-group label for {node_name} → {new_label_value}")
                logging.info(f"🏷️ Updated node-group label for {node_name} → {new_label_value}")
            else:
                print(f"✔️ Node {node_name} already labeled as node-group={new_label_value}")

            node_role = labels.get("node-role", "❌ Missing")
            #print(f"Node: {node_name} | node-role: {node_role}")
        print("======================================\n")
        logging.info("======================================\n")

    except Exception as e:
        print(f"❌ Error fetching or labeling node labels: {e}")
        logging.error(f"❌ Error fetching or labeling node labels: {e}")

def log_pod_locations(deployment_name, namespace):
    try:
        config.load_kube_config(config_file=k3s_config_file)
        core_v1_api = client.CoreV1Api()
        pods = core_v1_api.list_namespaced_pod(namespace)

        for pod in pods.items:
            if deployment_name in pod.metadata.name:
                print(f"📍 Pod {pod.metadata.name} is running on node: {pod.spec.node_name}")
                logging.info(f"📍 Pod {pod.metadata.name} is running on node: {pod.spec.node_name}")

    except Exception as e:
        print(f"❌ Error while checking pod placement: {e}")
        logging.error(f"❌ Error while checking pod placement: {e}")


def clean_to_number(value):
    """Ensure the value contains only numeric characters and supports floats."""
    # Allow only digits and a single decimal point
    cleaned_value = re.sub(r'[^\d.]', '', str(value))  # Remove all non-digit and non-dot characters
    if cleaned_value.count('.') > 1:  # Invalid if more than one decimal point
        raise ValueError("Please enter valid numeric values.")
    try:
        return float(cleaned_value)  # Convert to float
    except ValueError:
        raise ValueError("Please enter valid numeric values.")


# Function to handle user input and translate to JSON
def submit_intent():
    global membership_functions  # Access the global variable
    natural_language_input = ambiguous_input.get("1.0", tk.END).strip()

    if not natural_language_input:
        messagebox.showwarning("Warning", "Please provide a natural language input!")
        return

    collect_and_filter_kubernetes_data()
    generate_microservice_placement()
    label_nodes_to_file("k3s_cluster_info_important.json")  # update main
    shutil.copy("k3s_cluster_info_important.json", "before_cluster_snapshot.json")
    save_application_metrics("before_cluster_snapshot.json", "metrics_before_scaling.json")
    # 📂 Print the BEFORE Cluster Snapshot
    print("\n==================== BEFORE CLUSTER SNAPSHOT ====================")
    with open("before_cluster_snapshot.json", "r") as f:
        before_snapshot = json.load(f)
    print(json.dumps(before_snapshot, indent=4))  # Pretty print

    print("==================================================================\n")


    # Get thresholds for cost



    debug_print_node_costs()





    # Show a message to the user while processing
    messagebox.showinfo("Intent Submission", "Please wait for intent submission")


    # Create a combined input for GPT
    combined_input = (
        f"{natural_language_input}\n\n"
    )
    print("🟢 Final Combined Prompt Sent to GPT:")
    print(combined_input)

    # Call GPT to parse and clarify the intent
    parsed_intent = parse_with_gpt(combined_input)
    if not parsed_intent:
        return  # Exit if GPT failed to parse the intent

    #### Process and Save GPT Response
    try:

        # Save the intent to a JSON file
        with open("user_intent_fuzzy.json", "w") as file:
            json.dump(parsed_intent, file, indent=4)


        # Success message
        pretty_intent = json.dumps(parsed_intent, indent=4)  # Pretty format the JSON
        messagebox.showinfo(
            "Success",
            f"✅ Intent submitted and saved:\n\n{pretty_intent}\n\n"
            "Click 'OK' to confirm. Then click 'Evaluation' to view feedback, or submit a new intent if needed."
        )
        intent_submission_time= datetime.now()
        print(f"============== the intent is submitted on {intent_submission_time}")
        logging.info(f"=============== the intent is submitted on {intent_submission_time}")

        #---------------------------------------- to capture the intent time into a file
        # Append intent time and text to a log file
        with open("intent_timestamps.jsonl", "a") as f:
            f.write(json.dumps({
                "intent_submission_time": intent_submission_time.strftime("%Y-%m-%d %H:%M:%S"),
                "text": natural_language_input
            }) + "\n")
        #---------------------------------------- to capture the intent time into a file

        # Add to the GUI list
        existing_intents.append(parsed_intent)
        #update_intent_list()

        # Clear inputs
        #ambiguous_input.delete("1.0", tk.END)
        # 🚨 Reset Feedback (like resetting the text box)
        #feedback_label.config(text="")
        # 🚨 Reset Fuzzy Logic Details
        #fuzzy_text.delete("1.0", tk.END)
        # 🚨 Reset Fuzzy Rules
        #rules_text.delete("1.0", tk.END)

    except KeyError as e:
        print(f"KeyError: {e}")
        messagebox.showerror("Error", "Invalid response from GPT. Please try again.")
    except ValueError as e:
        print(f"ValueError: {e}")
        messagebox.showerror("Error", "Could not process details. Ensure GPT response is valid.")
    except Exception as e:
        print(f"Error saving intent: {e}")
        messagebox.showerror("Error", "An unexpected error occurred while saving the intent.")


# Function to call GPT for parsing ambiguous input
##################################################
def parse_with_gpt(natural_language_input):
    try:
        print("🟢 Starting GPT Call...")  # Debug Start
        logging.info("🟢 Starting GPT Call...")

        with open("fixed-files/fixed_prompt1.txt", "r") as file:
            fixed_prompt = file.read()

        messages = [
            {"role": "system", "content": fixed_prompt},
            {"role": "user", "content": natural_language_input}
        ]

        print("\n===== FULL PROMPT SENT TO GPT =====")
        logging.info("===== FULL PROMPT SENT TO GPT =====")
        for message in messages:
            print(f"{message['role'].upper()}: {message['content']}")
            logging.info(f"{message['role'].upper()}: {message['content']}")
        print("===================================")


        retry_count = 0
        MAX_RETRIES = 3  # Prevent infinite loops

        while retry_count < MAX_RETRIES:
            root.update()  # 🔄 Allow GUI updates
            print("🔵 Sending request to GPT...")  # Debug log before GPT call
            logging.info("🔵 Sending request to GPT...")

            response = openai.ChatCompletion.create(
                model="gpt-4o-2024-08-06",
                messages=messages,
                max_tokens=5000,  # Reduce token size
                temperature=0.7
            )

            # 🔵 Debug: Print the full raw GPT response
            #print("\n===== RAW GPT RESPONSE =====\n")
            #print(json.dumps(response, indent=4))  # Pretty-print the response
            #print("=================================\n")


            try:
                assistant_message = response["choices"][0]["message"]["content"]
                #print(f"🟠 GPT Response Received:\n{assistant_message}")

                # 🔴 Debug: Print just the extracted response
                #print("\n===== EXTRACTED GPT RESPONSE =====\n")
                #print(assistant_message)
                #print("=================================\n")

                # Remove Markdown artifacts (if present)
                if assistant_message.startswith("```json"):
                    assistant_message = assistant_message[7:-3]  # Remove ```json and ```

                # 🔵 Debug: Print cleaned JSON response before parsing
                #print("\n===== CLEANED JSON RESPONSE =====\n")
                #print(assistant_message)
                #print("=================================\n")

                # Try parsing JSON
                parsed_json = json.loads(assistant_message)
                print("✅ JSON Parsed Successfully")
                return parsed_json

            except json.JSONDecodeError:
                print("⚠️ JSON Parsing Failed - Response might be truncated!")
                logging.info("⚠️ JSON Parsing Failed - Response might be truncated!")
                # Retry with higher max_tokens if response is too short
                if len(assistant_message) < 100:
                    print("🔁 Retrying with a higher token limit...")
                    logging.info("🔁 Retrying with a higher token limit...")
                    return parse_with_gpt(natural_language_input)  # 🔁 Recursive retry

                user_response = simpledialog.askstring("GPT Needs Clarification", assistant_message)

                if not user_response:
                    messagebox.showwarning("Warning", "No response provided. Stopping.")
                    return None  # Stop if user cancels

                messages.append({"role": "assistant", "content": assistant_message})
                messages.append({"role": "user", "content": user_response})

        print("❌ Too many clarification requests. Exiting.")
        logging.info("❌ Too many clarification requests. Exiting.")
        messagebox.showerror("Error", "Too many clarification requests from GPT.")
        return None  # Stop after max retries

    except Exception as e:
        print(f"❌ Error in GPT Call: {e}")
        logging.error(f"❌ Error in GPT Call: {e}")
        messagebox.showerror("Error", f"Could not parse intent: {e}")
        return None


def get_crisp_input_from_intent(fuzzy_data, membership_functions):
    """
    Convert user intent levels into crisp input values using trapezoidal parameters.
    Formula: crisp_value = (param[1] + param[2]) / 2
    """
    crisp_values = {}
    print("\n")

    for var_name in ["response_time", "cost", "power", "reliability"]:
        user_level = fuzzy_data["intent"].get(var_name)
        if user_level not in membership_functions[var_name]:
            raise ValueError(f"❌ Invalid or missing {var_name} level: {user_level}")

        params = membership_functions[var_name][user_level]["parameters"]
        if isinstance(params, str):
            params = [float(p.strip()) for p in params.split(",")]

        crisp = (params[1] + params[2]) / 2
        crisp_values[var_name] = crisp


        print(f"✅ Mapped intent level for '{var_name} as '{user_level}' → crisp value: {crisp}")
        logging.info(f"✅ Mapped intent level for '{var_name} as '{user_level}' → crisp value: {crisp}")

    return crisp_values


def debug_fired_rules(rules_data, mf_dict, inputs):
    print("\n🔍 Debug: Fired Rules:")
    for idx, rule in enumerate(rules_data):
        rt_label = rule["if"]["response_time"]
        cost_label = rule["if"]["cost"]
        power_label = rule["if"]["power"]
        rel_label = rule["if"]["reliability"]
        rep_label = rule["then"]["scale_replicas"]
        place_label = rule["then"]["ratio_placement"]

        try:
            # Get trapezoidal parameters from the original membership_functions.json
            trapmf_params = {
                'response_time': mf_dict["response_time"][rt_label]["parameters"],
                'cost': mf_dict["cost"][cost_label]["parameters"],
                'power': mf_dict["power"][power_label]["parameters"],
                'reliability': mf_dict["reliability"][rel_label]["parameters"],
            }

            # Compute membership degrees
            firing_strengths = []
            for key, val in inputs.items():
                params = trapmf_params[key]
                degree = fuzz.trapmf(np.array([val]), params)[0]
                firing_strengths.append(degree)

            overall_strength = min(firing_strengths)
            if overall_strength > 0:
                print(f"🔥 Rule {idx + 1} fired with strength {overall_strength:.2f}:")
                print(f"    IF response_time IS {rt_label} AND cost IS {cost_label} "
                      f"AND power IS {power_label} AND reliability IS {rel_label}")
                print(f"    THEN scale_replicas IS {rep_label} AND ratio_placement IS {place_label}\n")

        except Exception as e:
            print(f"❌ Error while evaluating Rule {idx + 1}: {e}")



def run_fuzzy_logic():
    try:

        print_node_labels()
        # Step 1: Load user intent and membership functions
        with open("user_intent_fuzzy.json", "r") as file:
            fuzzy_data = json.load(file)
        with open("fixed-files/membership_functions.json", "r") as file:
            membership_functions = json.load(file)
        with open("fixed-files/fuzzy_rules.json", "r") as file:
            rules_data = json.load(file)

        # Step 2: Convert user intent to crisp input values
        crisp_inputs = get_crisp_input_from_intent(fuzzy_data, membership_functions)

        user_rt_level = fuzzy_data["intent"]["response_time"]
        user_cost_level = fuzzy_data["intent"]["cost"]
        user_power_level = fuzzy_data["intent"]["power"]
        user_reliability_level = fuzzy_data["intent"]["reliability"]

        # Step 3: Build the fuzzy system
        def extract_universe(mf_dict):
            vals = []
            for item in mf_dict.values():
                vals.extend(item["parameters"] if isinstance(item["parameters"], list) else [float(x) for x in
                                                                                             item["parameters"].split(
                                                                                                 ",")])
            return np.arange(min(vals), max(vals) + 0.1, 0.1)

        response_time = ctrl.Antecedent(extract_universe(membership_functions["response_time"]), 'response_time')
        cost = ctrl.Antecedent(extract_universe(membership_functions["cost"]), 'cost')
        power = ctrl.Antecedent(extract_universe(membership_functions["power"]), 'power')
        reliability = ctrl.Antecedent(extract_universe(membership_functions["reliability"]), 'reliability')
        replicas = ctrl.Consequent(extract_universe(membership_functions["scale_replicas"]), 'scale_replicas')
        placement = ctrl.Consequent(extract_universe(membership_functions["ratio_placement"]), 'ratio_placement')

        var_map = {
            "response_time": response_time,
            "cost": cost,
            "power": power,
            "reliability": reliability,
            "scale_replicas": replicas,
            "ratio_placement": placement
        }
        for var_name, sets in membership_functions.items():
            var = var_map[var_name]
            for label, details in sets.items():
                params = details["parameters"]
                if isinstance(params, str):
                    params = [float(p.strip()) for p in params.split(",")]
                var[label] = fuzz.trapmf(var.universe, params)

            # Debug fired rules
        debug_fired_rules(rules_data, membership_functions, crisp_inputs)

        # Step 4: Add rules and evaluate
        system = ctrl.ControlSystem()
        for rule in rules_data:
            antecedent = (
                    response_time[rule["if"]["response_time"]] &
                    cost[rule["if"]["cost"]] &
                    power[rule["if"]["power"]] &
                    reliability[rule["if"]["reliability"]]
            )
            system.addrule(ctrl.Rule(antecedent, replicas[rule["then"]["scale_replicas"]]))
            system.addrule(ctrl.Rule(antecedent, placement[rule["then"]["ratio_placement"]]))

        FS = ctrl.ControlSystemSimulation(system)
        FS.input["response_time"] = crisp_inputs["response_time"]
        FS.input["cost"] = crisp_inputs["cost"]
        FS.input["power"] = crisp_inputs["power"]
        FS.input["reliability"] = crisp_inputs["reliability"]
        FS.compute()

        # Step 5: Output and save
        scale_replicas = FS.output["scale_replicas"]
        ratio_placement = FS.output["ratio_placement"]

        #print("\n================ FINAL OUTPUT ================\n")
        #print(f"🔁 scale_replicas → {scale_replicas}")
        #print(f"📦 ratio_placement → {ratio_placement}")
        #print("==============================================\n")

        with open("fuzzy_output.json", "w") as f:
            json.dump({
                "new_replicas": scale_replicas,
                "new_placement": ratio_placement
            }, f, indent=4)
        #print("✅ Saved fuzzy_output.json")




        # 🧠 Capture pre-action state
        print("🟢 POD STATUS BEFORE SCALING")
        collect_and_filter_kubernetes_data(print_debug=True)



        # Save snapshot to separate files
        '''
        shutil.copy("application_consumed_cost.json", "before_action_consumed_cost.json")
        shutil.copy("application_consumed_power.json", "before_action_consumed_power.json")
        shutil.copy("application_reliability.json", "before_action_reliability.json")
        '''

        # call take action function
        take_action()

        # ✅ Display the result

        print("========================== FINAL OUTPUT ==========================")
        print(f"🔁 Recommended scale_replicas → {scale_replicas}")
        logging.info(f"🔁 Recommended scale_replicas → {scale_replicas}")

        # ------ interpret the scale_replicas- start
        percentage_change_for_scale_replicas = (scale_replicas - 0.5) * 2 * 100
        # ✅ Interpret and print scaling direction
        if percentage_change_for_scale_replicas > 0:
            print(f"🧠 Scale UP by {percentage_change_for_scale_replicas:.1f}%")
            logging.info(f"🧠 Scale UP by {percentage_change_for_scale_replicas:.1f}%")
        elif percentage_change_for_scale_replicas < 0:
            print(f"🧠 Scale DOWN by {abs(percentage_change_for_scale_replicas):.1f}%")
            logging.info(f"🧠 Scale DOWN by {abs(percentage_change_for_scale_replicas):.1f}%")
        else:
            print("🧠 No scaling required (0%)")
            logging.info("🧠 No scaling required (0%)")
        # ------ interpret the scale_replicas-end

        print(f'📦 Recommended ratio_placement →  {ratio_placement}')
        logging.info(f'📦 Recommended ratio_placement →  {ratio_placement}')

        # ------ interpret the ratio_placement- start
        cloud_percentage = round(ratio_placement * 100)
        edge_percentage = 100 - cloud_percentage
        print(f"🧠 Placement ratio interpreted → {cloud_percentage}% pods on cloud, {edge_percentage}% pods on edge.")
        logging.info(
            f"🧠 Placement ratio interpreted → {cloud_percentage}% pods on cloud, {edge_percentage}% pods on edge.")
        print("====================================================================")



        # ------ interpret the ratio_placement- end

        # ---- for feedback-----

        wait_for_pods_ready()
        print("🟢 POD STATUS AFTER SCALING")
        collect_and_filter_kubernetes_data(print_debug=True)
        label_nodes_to_file("after_cluster_snapshot.json")
        save_application_metrics("after_cluster_snapshot.json", "metrics_after_scaling.json")

        compare_metrics(
            before_file="metrics_before_scaling.json",
            after_file="metrics_after_scaling.json",
            intent_file="user_intent_fuzzy.json",
            output_file="comparison_report.json"
        )
        plot_comparison_graphs_in_gui("comparison_report.json")
        verify_intent_satisfaction_from_traffic()
        #plot_comparison_graphs("comparison_report.json")



        # 📂 Print the AFTER Cluster Snapshot
        #print("\n==================== AFTER CLUSTER SNAPSHOT =====================")
        #with open("after_cluster_snapshot.json", "r") as f:
            #after_snapshot = json.load(f)
        #print(json.dumps(after_snapshot, indent=4))  # Pretty print

        #print("==================================================================\n")





        # ✅ Update GUI Feedback Label to show the outcome based on user's intent
        feedback_message = (
            f"✔️ Congratulations! Your intent has been successfully satisfied!\n"
            f"User requested {user_rt_level} response time and {user_cost_level} cost.\n"
            f"User also preferred {user_power_level} power usage and {user_reliability_level} reliability.\n"
            f"• To satisfy {user_rt_level} response time, the system configured {scale_replicas} for scale_replicas.\n"
            f"• To satisfy {user_cost_level} cost, the system used {ratio_placement} for ratio_placement strategy."
        )
        feedback_label.config(
            text=feedback_message,
            fg="green",
            wraplength=600,  # Already used, but keep it here for safety
            justify="left"
        )


    #------creating combined file
        # ✅ Create finalized.json here
        try:
            with open("user_intent_fuzzy.json", "r") as f:
                intent_data = json.load(f)
            with open("metrics_before_scaling.json", "r") as f:
                before_metrics = json.load(f)
            with open("metrics_after_scaling.json", "r") as f:
                after_metrics = json.load(f)
            with open ("before_cluster_snapshot.json", "r") as f:
                before_snapshot = json.load(f)
            with open ("after_cluster_snapshot.json", "r") as f:
                after_snapshot = json.load(f)
            with open ("fuzzy_output.json", "r") as f:
                fuzzy_result= json.load(f)



            combined = {
                "user_intent": intent_data,
                "metrics_before_scaling": before_metrics,
                "metrics_after_scaling": after_metrics,
                "before_cluster_snapshot": before_snapshot,
                "after_cluster_snapshot": after_snapshot,
                "fuzzy_output": fuzzy_result

            }

            with open("finalized.json", "w") as f:
                json.dump(combined, f, indent=4)

            print("📁 Combined finalized.json file created successfully.")
            logging.info("📁 Combined finalized.json file created successfully.")

        except Exception as e:
            print(f"❌ Error creating finalized.json: {e}")
            logging.error(f"❌ Error creating finalized.json: {e}")


    # ------creating combined file


    except Exception as e:
        print(f"❌ Error running fuzzy logic: {e}")
        logging.error(f"❌ Error running fuzzy logic: {e}")


# Function to add placeholder text to an Entry widget
def add_placeholder(entry, placeholder):
    entry.insert(0, placeholder)
    entry.config(fg="grey")

    def on_focus_in(event):
        if entry.get() == placeholder:
            entry.delete(0, tk.END)
            entry.config(fg="black")

    def on_focus_out(event):
        if not entry.get():
            add_placeholder(entry, placeholder)

    entry.bind("<FocusIn>", on_focus_in)
    entry.bind("<FocusOut>", on_focus_out)


def plot_fuzzy_membership(membership_functions):
    if not membership_functions:
        messagebox.showwarning("Warning", "No membership functions available to display.")
        return
    #categories = ["response_time", "cost", "replicas", "placement"]  # Include all three categories
    categories = list(membership_functions.keys())

    for category in categories:
        fuzzy_sets = membership_functions.get(category, {})
        if not fuzzy_sets:
            continue  # Skip if no membership functions available

        # Create a new figure for each category
        plt.figure(figsize=(8, 4))
        for set_name, details in fuzzy_sets.items():
            params = details["parameters"]
            if details["type"] == "triangular":
                x = [params[0], params[1], params[2]]
                y = [0, 1, 0]
                plt.plot(x, y, label=set_name.capitalize())
            elif details["type"] == "trapezoidal":
                x = [params[0], params[1], params[2], params[3]]
                y = [0, 1, 1, 0]
                plt.plot(x, y, label=set_name.capitalize())

        # Plot details
        plt.title(f"Fuzzy Membership Functions for {category.capitalize()}")
        plt.xlabel(category.capitalize())
        plt.ylabel("Membership Value")
        plt.grid(True)
        plt.legend(loc="upper right")

        # Display the plot
        plt.tight_layout()
        plt.show()

#============================GUI and requirements===================================
# Main application window
root = tk.Tk()
root.title("Intent Management System")
root.geometry("750x800")
root.config(bg="white")


# Human Image on Top Left
try:
    human_img = Image.open("fixed-files/intent.png")
    human_img = human_img.resize((110, 110), Image.Resampling.LANCZOS)
    human_photo = ImageTk.PhotoImage(human_img)
    human_image_label = tk.Label(root, image=human_photo, bg="white")
    human_image_label.place(x=10, y=10)
except Exception as e:
    messagebox.showwarning("Warning", f"Could not load human image: {e}")

# User Image on Top Right
try:
    user_img = Image.open("fixed-files/user.jpeg")
    user_img = user_img.resize((110, 110), Image.Resampling.LANCZOS)
    user_photo = ImageTk.PhotoImage(user_img)
    user_image_label = tk.Label(root, image=user_photo, bg="white")
    user_image_label.place(x=630, y=10)
except Exception as e:
    messagebox.showwarning("Warning", f"Could not load user image: {e}")

# Header Text
header_text = tk.Label(root, text="Please define the intent below \n in the natural language format and press submit", font=("Arial", 15, "bold"), bg="white", fg="#333")
header_text.pack(pady=10)

# Input Frame
input_frame = tk.Frame(root, bg="white")
input_frame.pack(pady=20, anchor='center')

#>>>>>>>>>>>>>>>>>>>>> for the user feedback

# Create a Frame for User Feedback (Box around the text)
feedback_frame = tk.Frame(root, bg="lightgray", bd=2, relief="groove")  # Border and relief for the box
feedback_frame.pack(pady=10, fill="x", padx=50)  # Add padding and centering

# Add Title for the Feedback Box
tk.Label(feedback_frame, text="User Feedback", font=("Arial", 12, "bold"), fg= 'darkblue', bg="lightblue").pack(pady=5)

# Move the Intent Satisfaction Status Label inside the Feedback Box
feedback_label = tk.Label(
    feedback_frame,
    text="Feedback will appear here after evaluation... \n",

    font=("Arial", 12),
    fg="red",
    bg="lightgray",
    wraplength=600,
    justify="center"  # Center text inside the box
)
feedback_label.pack(pady=5)
#>>>>>>>>>>>>>>>>>>>> for the user feedback


# Ambiguous Input Text Box
#tk.Label(input_frame, text="Natural Language Intent:", font=("Arial", 12), bg="white").grid(row=0, column=0, columnspan=2, pady=5, sticky="n")
ambiguous_input = tk.Text(input_frame, width=30, height=4, font=("Arial", 12))
ambiguous_input.grid(row=2, column=1, padx=10, pady=10)

# Submit Button
submit_button = tk.Button(
    input_frame,
    text="Submit",
    command=submit_intent,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=20,
    relief="groove",
    borderwidth=2
)
submit_button.grid(row=3, column=0, columnspan=2, pady=20)
submit_button.config(highlightbackground="#4CAF50", highlightcolor="#4CAF50")

# Add Fuzzy Logic Details Section
fuzzy_frame = tk.Frame(root, bg="white")
fuzzy_frame.pack(pady=2, anchor='center')  # Reduced padding and width expansion

tk.Label(fuzzy_frame, text="Fuzzy Logic Details:", font=("Arial", 12, "bold"), bg="white").pack(pady=2, anchor='center')


# Scrollbar for Fuzzy Details
fuzzy_scroll = tk.Scrollbar(fuzzy_frame, orient="vertical")
fuzzy_text = tk.Text(fuzzy_frame, height=1, font=("Arial", 12), wrap="word", yscrollcommand=fuzzy_scroll.set)
fuzzy_text.configure(height=4)  # Explicitly force small height
fuzzy_scroll.config(command=fuzzy_text.yview)

fuzzy_text.pack(side="left", expand=False, padx=5, pady=2)
fuzzy_scroll.pack(side="right", fill="y")


# Add Fuzzy Rules Section
rules_frame = tk.Frame(root, bg="white")
rules_frame.pack(pady=4, anchor='center')  # Reduced padding and width expansion

tk.Label(rules_frame, text="Fuzzy Rules:", font=("Arial", 12, "bold"), bg="white").pack(pady=2, anchor='center')

# Scrollbar for Fuzzy Rules
rules_scroll = tk.Scrollbar(rules_frame, orient="vertical")
rules_text = tk.Text(rules_frame, height=4, font=("Arial", 12), wrap="word", yscrollcommand=rules_scroll.set)
rules_text.configure(height=4)  # Explicitly force small height
rules_scroll.config(command=rules_text.yview)

rules_text.pack(side="left", expand=False, padx=5, pady=2)
rules_scroll.pack(side="right", fill="y")

# Keep the Fuzzy Graph Button Separate
fuzzy_graph_button = tk.Button(
    root,
    text="Show Fuzzy Graph",
    command=lambda: plot_fuzzy_membership(load_membership_functions()),
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=8,
    pady=4,
    relief="groove"
)
fuzzy_graph_button.pack(pady=10, anchor='center')  # Reduced padding

# Data storage
existing_intents = []


# Add Button to Trigger Real-Time Fuzzy Logic
real_time_button = tk.Button(
    input_frame,
    text="Evaluation",
    command=run_fuzzy_logic,
    bg="#4CAF50",
    fg="white",
    font=("Arial", 12, "bold"),
    padx=20,
    relief="groove",
    borderwidth=2
)
real_time_button.grid(row=6, column=0, columnspan=2, pady=20)

#---helper for plotting
def load_membership_functions():
    try:
        with open("fixed-files/membership_functions.json", "r") as f:
            return json.load(f)
    except Exception as e:
        print(f"❌ Could not load membership functions: {e}")
        return None


def debug_print_node_costs():
    try:
        with open("k3s_cluster_info_important.json", "r") as file:
            data = json.load(file)

        nodes = data["Kubernetes"]["nodes"]
        pods = data["Kubernetes"]["pods"]

        node_capacity = {}
        for node in nodes:
            cpu_str = node["capacity"]["cpu"]
            mem_str = node["capacity"]["memory"]
            cpu = float(cpu_str.replace("m", "")) / 1000 if "m" in cpu_str else float(cpu_str)
            mem = float(mem_str.replace("Ki", "")) / 1024  # Convert to Mi
            node_capacity[node["name"]] = {"cpu": cpu, "memory": mem}

        node_usage = {}
        for pod in pods:
            if not pod["name"].startswith("microservice"):
                continue

            node_name = pod["node"]
            cpu_str = pod["resources"]["cpu_limit"]
            mem_str = pod["resources"]["memory_limit"]
            cpu = float(cpu_str.replace("m", "")) / 1000 if "m" in cpu_str else float(cpu_str)
            mem = float(mem_str.replace("Mi", ""))  # Already in Mi

            if node_name not in node_usage:
                node_usage[node_name] = {
                    "cpu": 0.0,
                    "memory": 0.0,
                    "pods": []
                }
            ##
            node_usage[node_name]["cpu"] += cpu
            node_usage[node_name]["memory"] += mem
            node_usage[node_name]["pods"].append({
                "name": pod["name"],
                "cpu": cpu,
                "memory": mem
            })


        total_end_to_end_cost = 0.0
        print("\n🔍 Per-Node Cost Breakdown (with Pod Details):")
        for node_name, usage in node_usage.items():
            capacity = node_capacity.get(node_name)
            if not capacity:
                continue

            #
            cpu_percent = (usage["cpu"] / capacity["cpu"]) * 100
            mem_percent = (usage["memory"] / capacity["memory"]) * 100
            avg_cost = round((cpu_percent + mem_percent) / 2, 2)
            total_end_to_end_cost += avg_cost

            print(f"\n🖥️ Node: {node_name}")
            print(f"   ➤ CPU Used: {usage['cpu']} / {capacity['cpu']} cores ({cpu_percent:.2f}%)")
            print(f"   ➤ MEM Used: {usage['memory']} / {capacity['memory']} MiB ({mem_percent:.2f}%)")
            print(f"   ➤ Average Cost: {avg_cost}%")
            print(f"   ➤ Pods running on this node:")
            for pod in usage["pods"]:
                print(f"     • {pod['name']}: CPU={pod['cpu']} cores, MEM={pod['memory']} MiB")

            # Optional: Print total end-to-end cost


        print(f"\n🧮 Total End-to-End Cost Across Nodes: {round(total_end_to_end_cost, 2)}% \n")
    except Exception as e:
        print(f"❌ Error in debug_print_node_costs: {e}")


#---- waiting function for pods to become ready
def wait_for_pods_ready(timeout=120, check_interval=5):
    """
    Waits until all microservice pods are in 'Running' state and ready.
    """
    from kubernetes.client.rest import ApiException

    print("⏳ Waiting for all microservice pods to be ready...")
    logging.info("⏳ Waiting for all microservice pods to be ready...")

    config.load_kube_config(config_file=k3s_config_file)
    v1 = client.CoreV1Api()

    start_time = time.time()

    while time.time() - start_time < timeout:
        try:
            pod_list = v1.list_pod_for_all_namespaces()
            microservice_pods = [
                pod for pod in pod_list.items
                if pod.metadata.name.startswith("microservice")
            ]

            all_ready = True
            for pod in microservice_pods:
                # Check pod phase
                if pod.status.phase != "Running":
                    all_ready = False
                    break

                # Check container statuses
                conditions = pod.status.conditions or []
                ready_condition = next((c for c in conditions if c.type == "Ready"), None)
                if not ready_condition or ready_condition.status != "True":
                    all_ready = False
                    break

            if all_ready:
                print("✅ All microservice pods are running and ready.")
                logging.info("✅ All microservice pods are running and ready.")
                return True

        except ApiException as e:
            print(f"❌ Error checking pod readiness: {e}")
            logging.error(f"❌ Error checking pod readiness: {e}")
            return False

        time.sleep(check_interval)

    print("❌ Timeout: Not all pods became ready in time.")
    logging.warning("❌ Timeout: Not all pods became ready in time.")
    return False


def debug_print_pod_statuses(pods_data):
    print("\n🔍 Debugging Pod Statuses:")
    for pod in pods_data:
        print(f"📦 Pod: {pod['name']} on Node: {pod['node']}")
    print("✅ All pods above are Running + Ready (after filtering)\n")

#-----------------Evaluation
def compare_metrics(before_file="metrics_before_scaling.json", after_file="metrics_after_scaling.json", intent_file="user_intent_fuzzy.json", output_file="comparison_report.json"):
    try:
        # Load metrics and user intent
        with open(before_file, "r") as f:
            before_metrics = json.load(f)

        with open(after_file, "r") as f:
            after_metrics = json.load(f)

        with open(intent_file, "r") as f:
            user_intent_data = json.load(f)

        user_intent = user_intent_data.get("intent", {})

        comparison_result = {}

        print("\n🔵 Comparison Report Based on User Intent:\n")

        for key in before_metrics:
            before_value = before_metrics[key]
            after_value = after_metrics.get(key)

            # Skip if the metric is missing in after file
            if after_value is None:
                print(f"⚠️ Metric '{key}' missing in after-scaling file.")
                continue

            entry = {
                "before": round(before_value, 2),
                "after": round(after_value, 2)
            }

            if before_value == 0:
                print(f"🔸 {key}: Before = 0, cannot calculate improvement.")
                entry["improvement_percentage"] = None
                entry["satisfied"] = False
            else:
                change_percentage = ((after_value - before_value) / abs(before_value)) * 100
                change_percentage = round(change_percentage, 2)
                entry["improvement_percentage"] = change_percentage

                # Determine which intent field this metric belongs to
                if key in ["application_consumed_cost", "application_monetary_cost"]:
                    intent_goal = user_intent.get("cost", "low")  # default to low
                elif key == "application_consumed_power":
                    intent_goal = user_intent.get("power", "low")
                elif key == "application_reliability":
                    intent_goal = user_intent.get("reliability", "high")
                else:
                    intent_goal = "low"  # Default behavior if unknown metric

                # Now, determine if Satisfied based on the intent goal
                if intent_goal == "low":
                    satisfied = after_value <= before_value + (abs(before_value) * 0.01)  # allow small increase (<1%)
                elif intent_goal == "high":
                    satisfied = after_value >= before_value - (abs(before_value) * 0.01)  # allow small decrease (<1%)
                else:
                    satisfied = False  # unexpected

                entry["satisfied"] = satisfied

                # -------- Printing per metric
                print(f"📊 {key.replace('_', ' ').title()}:")
                print(f"   - Before: {round(before_value, 2)}")
                print(f"   - After: {round(after_value, 2)}")
                if entry["improvement_percentage"] is not None:
                    print(f"   - Change: {abs(change_percentage):.2f}%")
                    print(f"   - Intent Goal: {intent_goal.upper()}")
                    print(f"   - Status: {'✅ Satisfied' if satisfied else '❌ Not Satisfied'}")
                print()

            comparison_result[key] = entry

        # Save the full report
        with open(output_file, "w") as f:
            json.dump(comparison_result, f, indent=4)

        print(f"✅ Full comparison report saved to '{output_file}'.\n")

    except Exception as e:
        print(f"❌ Error in compare_metrics: {e}")


def plot_comparison_graphs(comparison_file="comparison_report.json"):
    try:
        # Load comparison results
        with open(comparison_file, "r") as f:
            comparison_result = json.load(f)

        # 🛠️ Custom name mapping for cleaner chart labels
        metric_name_mapping = {
            "application_consumed_cost": "Consumed_Cost(%)",
            #"application_monetary_cost": "Node_Cost($/h)",
            "application_consumed_power": "Consumed_Power(W)",
            "application_reliability": "Reliability(Probability)"
        }

        metrics = []
        before_values = []
        after_values = []
        satisfaction = {"satisfied": 0, "unsatisfied": 0}

        for metric, data in comparison_result.items():
            metrics.append(metric_name_mapping.get(metric, metric))
            before_values.append(data.get("before", 0))
            after_values.append(data.get("after", 0))

            if data.get("satisfied", False):
                satisfaction["satisfied"] += 1
            else:
                satisfaction["unsatisfied"] += 1

        # ---------- Plot Grouped Bar Chart ----------
        x = range(len(metrics))  # Positions for metrics
        width = 0.35  # Bar width

        plt.figure(figsize=(12, 6))
        plt.bar([pos - width/2 for pos in x], before_values, width=width, label="Before Scaling", color="skyblue")
        plt.bar([pos + width/2 for pos in x], after_values, width=width, label="After Scaling", color="lightgreen")

        plt.xlabel("Metrics", fontsize=14)
        plt.ylabel("Values", fontsize=14)
        plt.title("Before vs After Scaling Comparison", fontsize=16, fontweight='bold')
        plt.xticks(x, metrics, rotation=10, ha="right")
        plt.legend()
        plt.grid(axis="y", linestyle="--", alpha=0.7)
        plt.tight_layout()
        plt.show()

        # ---------- Plot Satisfaction Pie Chart ----------
        labels = ['Satisfied', 'Unsatisfied']
        sizes = [satisfaction["satisfied"], satisfaction["unsatisfied"]]
        colors = ['#4CAF50', '#F44336']  # Green and Red

        plt.figure(figsize=(6, 6))
        plt.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
        plt.title("Intent Satisfaction Summary", fontsize=16, fontweight='bold')
        plt.tight_layout()
        plt.show()

    except Exception as e:
        print(f"❌ Error in plot_comparison_graphs: {e}")

def plot_comparison_graphs_in_gui(comparison_file="comparison_report.json"):
    try:
        # Load comparison results
        with open(comparison_file, "r") as f:
            comparison_result = json.load(f)

        # 🛠️ Custom name mapping for cleaner chart labels
        metric_name_mapping = {
            "application_consumed_cost": "Consumed_Cost(%)",
            "application_monetary_cost": "Node_Cost($/h)",
            "application_consumed_power": "Consumed_Power(W)",
            "application_reliability": "Reliability(Probability)"
        }

        metrics = []
        before_values = []
        after_values = []
        satisfaction = {"satisfied": 0, "unsatisfied": 0}

        for metric, data in comparison_result.items():
            metrics.append(metric_name_mapping.get(metric, metric))
            before_values.append(data.get("before", 0))
            after_values.append(data.get("after", 0))

            if data.get("satisfied", False):
                satisfaction["satisfied"] += 1
            else:
                satisfaction["unsatisfied"] += 1

        # Create a popup window
        popup = tk.Toplevel()
        popup.title("Comparison Charts")
        popup.geometry("1200x800")

        # Create two figures: one for bar chart, one for pie chart
        fig_bar, ax_bar = plt.subplots(figsize=(8, 4))
        fig_pie, ax_pie = plt.subplots(figsize=(5, 5))

        # --- Bar Chart ---
        x = range(len(metrics))
        width = 0.35
        ax_bar.bar([pos - width/2 for pos in x], before_values, width=width, label="Before Scaling", color="skyblue")
        ax_bar.bar([pos + width/2 for pos in x], after_values, width=width, label="After Scaling", color="lightgreen")
        ax_bar.set_xlabel("Metrics")
        ax_bar.set_ylabel("Values")
        ax_bar.set_title("Before vs After Scaling Comparison", fontweight='bold')
        ax_bar.set_xticks(list(x))
        ax_bar.set_xticklabels(metrics, rotation=10, ha="right")
        ax_bar.legend()
        ax_bar.grid(axis="y", linestyle="--", alpha=0.7)

        # --- Pie Chart ---
        labels = ['Satisfied', 'Unsatisfied']
        sizes = [satisfaction["satisfied"], satisfaction["unsatisfied"]]
        colors = ['#4CAF50', '#F44336']
        ax_pie.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=140, wedgeprops={'edgecolor': 'black'})
        ax_pie.set_title("Intent Satisfaction Summary", fontweight='bold')

        # --- Embed charts into the popup window ---
        canvas_bar = FigureCanvasTkAgg(fig_bar, master=popup)
        canvas_bar.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        canvas_pie = FigureCanvasTkAgg(fig_pie, master=popup)
        canvas_pie.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Draw both plots
        canvas_bar.draw()
        canvas_pie.draw()
    except Exception as e:
        print(f"❌ Error in plot_comparison_graphs_in_gui: {e}")

def verify_intent_satisfaction_from_traffic(
    fuzzy_file="fuzzy_output.json",
    bounds_file="fixed-files/replica_bounds.json",
    cluster_file="before_cluster_snapshot.json",

):
    try:


        with open(fuzzy_file, "r") as f:
            fuzzy = json.load(f)
        with open(bounds_file, "r") as f:
            bounds = json.load(f)
        with open(cluster_file, "r") as f:
            cluster = json.load(f)

        # Read current traffic from Locust
        response = requests.get(f"http://{locust_master_url}:8089/stats/requests")
        response.raise_for_status()
        user_count = response.json().get("user_count", 0)

        print(f"\n👥 Current Locust Users: {user_count}")
        print("🔎 Verifying replica scale and placement per service...")

        scale_ratio = float(fuzzy.get("new_replicas", 1))
        placement_ratio = float(fuzzy.get("new_placement", 0.5))

        deployments = cluster["Kubernetes"]["deployments"]

        satisfied = True

        for dep in deployments:
            dep_name = dep["name"]
            if "-cloud" in dep_name or "-edge" in dep_name:
                base_name = dep_name.rsplit("-", 1)[0]
            else:
                base_name = dep_name
            ms_match = re.match(r"(microservice\d+)", base_name)
            if not ms_match:
                continue

            ms = ms_match.group(1)
            level_keys = list(bounds[ms].keys())
            closest_level = min(level_keys, key=lambda k: abs(user_count - int(k)))
            min_r = bounds[ms][closest_level]["min"]
            max_r = bounds[ms][closest_level]["max"]
            expected_replicas = round(min_r + scale_ratio * (max_r - min_r))

            # Count actual replicas
            edge_reps = sum(d["replicas"] for d in deployments if f"{ms}-deployment-edge" == d["name"])
            cloud_reps = sum(d["replicas"] for d in deployments if f"{ms}-deployment-cloud" == d["name"])
            total = edge_reps + cloud_reps
            cloud_expected = round(expected_replicas * placement_ratio)
            edge_expected = expected_replicas - cloud_expected

            ok_replicas = (total == expected_replicas)
            ok_placement = (cloud_reps == cloud_expected and edge_reps == edge_expected)

            print(f"\n📦 {ms}: Expected {expected_replicas} → Cloud: {cloud_expected}, Edge: {edge_expected}")
            print(f"   - Actual: Cloud={cloud_reps}, Edge={edge_reps}, Total={total}")
            print(f"   - Replica Match: {'✅' if ok_replicas else '❌'}")
            print(f"   - Placement Match: {'✅' if ok_placement else '❌'}")

            if not ok_replicas or not ok_placement:
                satisfied = False


        print("\n🎯 Final Intent Satisfaction:", "✅ SATISFIED" if satisfied else "❌ NOT SATISFIED")
        return satisfied

    except Exception as e:
        print(f"❌ Error in verifying intent satisfaction: {e}")



take_action_lock = threading.Lock()
last_action_time = 0
cooldown_seconds = 100  # 3 minutes cooldown


def scheduler():
    global last_action_time
    try:
        now = time.time()

        # Prevent concurrent execution and enforce cooldown
        with take_action_lock:
            if now - last_action_time < cooldown_seconds:
                print(
                    f"⏳ Cooldown in effect. Skipping take_action. Time left: {cooldown_seconds - (now - last_action_time):.1f}s")
                return

            last_action_time = now  # update the time only if allowed to proceed

            print("🟢 Scheduler triggered based on Locust traffic change.")

            # Step 1: Create fuzzy_output.json only if it does not exist
            fuzzy_output_path = "fuzzy_output.json"
            if not os.path.exists(fuzzy_output_path):
                fuzzy_output = {
                    "new_replicas": 0.5,
                    "new_placement": 0.5
                }
                with open(fuzzy_output_path, "w") as f:
                    json.dump(fuzzy_output, f, indent=4)
                print("Created fuzzy_output.json with default values.")
            else:
                print("fuzzy_output.json already exists. Skipping creation.")

            # Step 2: Take action using existing logic
            print(
                '##############################\n take_action started inside the scheduler \n ##############################\n')
            take_action()
            print(
                ' \n ##############################\n take_action ended inside the scheduler \n ########################## \n')

    except Exception as e:
        print(f"❌ Error in scheduler: {e}")


def poll_user_level_change(interval=20, replica_bounds_file="fixed-files/replica_bounds.json"):
    try:
        with open(replica_bounds_file, "r") as f:
            bounds = json.load(f)

        # Extract available user levels (e.g., [1, 5, 10, ...])
        #defined_levels = sorted(set(int(k) for k in bounds["microservice1"].keys()))
        defined_levels = [1, 10, 20, 30, 40]
        current_level = None

        def poll():
            nonlocal current_level
            while True:
                try:
                    user_count = get_current_user_count()
                    if user_count in defined_levels and user_count != current_level:
                        print(f"🔄 Detected exact Locust user level change: {current_level} → {user_count}")
                        current_level = user_count
                        scheduler()


                except Exception as e:
                    print(f"❌ Error in user level polling: {e}")

                time.sleep(interval)

        # Run in background thread
        polling_thread = threading.Thread(target=poll, daemon=True)
        polling_thread.start()
        print("📡 Locust user-level polling started.")

    except Exception as e:
        print(f"❌ Error initializing user-level polling: {e}")

poll_user_level_change(interval=10)

# Run the application
root.mainloop()
