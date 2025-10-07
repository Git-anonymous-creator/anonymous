Microservice1_url = 'http://10.43.138.102:5001'   # no need to put endpoint "/resize" here
webhooks = 'http://10.43.135.8:5002/bw,http://10.43.107.4:8081,http://10.43.6.198:5004/notify'
central_db_url = 'http://10.43.7.90:5005/track_time'
logs_url = 'http://10.43.7.90:5005/log' #log endpoint in DB
db_url_get_time = 'http://10.43.7.90:5005/get_time'








alpha = 0.02
edge_cost_value= (1,5)
cloud_cost_value = (20,30)

edge_probability_of_failure = (0.01,0.10) # high failure range for the edge nodes
cloud_probability_of_failure =(0.001,0.005) # low failure range for the cloud nodes

#========cpu and mem price
cloud_cpu_core_price= 0.0058 # per core per hour
cloud_mem_GB_price = 0.000005664 # per MiB per hour: 0.0058 / 1024

edge_cpu_core_price = 0.00116 # edge: 80% cheaper
edge_mem_GB_price = 0.000001134 #  edge: 80% cheaper
#================================

#for power==========
cloud_nodes_idle= 6.2 #Watts
cloud_nodes_max= 21.0 #Watts

edge_nodes_idle= 4.96 #Watts
edge_nodes_max= 16.8  #Watts
#====================

#---------------------
min_rt=2
max_rt=10
#---------------------
crisp_value_for_placement = 5



probability_of_failure = (0.01,0.5) # generating random values for node failure


min_range_reliability= 0.01
max_range_reliability= 0.05

node_cost_range_random = (10, 20) # unit is $- eg:10$ per hour

P_MAX = 200  # watts per node at full load
P_IDLE = 50
reliability_range= (1,5)

for_overlap_ratio = 0.05

master_ip_address = '10.0.0.100'
worker1_ip_address = '10.0.0.101'
worker2_ip_address = '10.0.0.102'
worker3_ip_address = '10.0.0.103'
openai_api_key = "your_openai_api_key_here"
k3s_config_file = "/etc/rancher/k3s/k3s.yaml"


locust_master_url='118.138.236.63'
#locust_master_url='118.138.238.4'
onos_ip_address = '118.138.238.60:8181'
sflow_rt_ip_address = '118.138.238.60:8008'
Prometheus_ip_address = '118.138.238.60:9090'
json_path_directory = '/home/ubuntu/PycharmProjects/pythonProject/my_work/files/json/'
others_path_directory = '/home/ubuntu/PycharmProjects/pythonProject/my_work/files/others/'
csv_path_directory = '/home/ubuntu/PycharmProjects/pythonProject/my_work/files/csv/'



##############################################################################################################################
##############################################################################################################################
# node_cost($/h): $$$ per H for each node- randomly assigned a $$/H to each node

# node_power (Watts): node_power = P_IDLE + (P_MAX - P_IDLE) * cpu_ratio
#               P_IDLE = 50 (default idle power in watts)
#               P_MAX = 200 (default max power in watts)
#               cpu_ratio = used_cpu / total_cpu
#               This is a linear model:
#                       At 0% CPU usage → node uses 50W
#                       At 100% CPU usage → node uses 200W
#
#
# consumed_cost(%): application_consumed_cost (USD $/hr)
#                       cpu_usage_percent for each pod= (pod_cpu_cores / node_total_cpu) × 100
#                       memory_usage_percent for each pod= (pod_memory_limit/ node_total_memory) × 100
#                       x= cpu_price_per_core_hour (different fro edge and cloud)- edge is 80% cheaper
#                       y= mem_price_per_gb_hour (different fro edge and cloud)- edge is 80% cheaper

#                       cost_Rate for each pod= [cpu_usage_percent *x]+ [memory_usage_percent* y]
#                       cost_Rate for the entire application= sum of  cost_Rate for all pod )= p1+p2+p3+p4
#
#

#                   application_monetary_cost: (dollar per hour)
#                            The amount of money (dollar) we need to pay for running this application on the nodes per hours)
#                            total of ([cpu_usage/100]* node_cost) on all nodes
#
# consumed_power(Watts): application_consumed_power = sum(consumed_power of all nodes)
#                       Each pod's contribution to power is proportional to its CPU request vs. node’s total CPU capacity.
#                       All pod contributions are summed for the node.
#                           per node:
#                               consumed_power = sum_over_pods( node_power * (pod_cpu / node_total_cpu) )
#
#
# application_reliability(Unitless Probability (Value between 0.01 and 0.5):
#                        application_reliability = Π (microservice_reliability for all microservices)
#                        A series system: If any microservice fails, the whole application is affected.
#                        A microservice may tolerate a node failure if it has replicas on multiple nodes.
#                               per microservice:
#                                   failure_product = Π (node_probability_of_failure for all nodes running replicas of the microservice)
#                                   microservice_reliability = 1 - failure_product

#unit>>>> probability
############################################################################################################################################
##############################################################################################################################