#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import sys
import signal

import subprocess
import logging
import asyncio
import websockets
import argparse
from pathlib import Path

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     L O C A L     L I B R A R I E S   /   F I L E S                             #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters.                                                                   #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
PYTHON_PATH = Path(sys.executable).name
S_FILE_PATH = Path(__file__).resolve().parents[0].joinpath("run_websocket_worker.py")
PROCESS_LIST = []

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   starting websocket server for a given client / worker node. Each now worker has a server.   #
#                                                                                               #
#***********************************************************************************************#
async def send_local_info(remote_host, remote_port, worker_list):
    # send local client information
    uri = 'ws://{0}:{1}'.format(remote_host, remote_port)

    # print log message
    print("establishing connection at {0}".format(uri))
    
    # send information of each worker to the server
    for worker in worker_list:
        async with websockets.connect(uri) as websocket:
            await websocket.send(worker[2])
            await websocket.send(worker[0])
            await websocket.send("{0}".format(worker[1]))

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper function to create a list of workers assuming consecutive ports are avialable.       #
#                                                                                               #
#***********************************************************************************************#
def generate_worker_list(suffix_id, worker_host, starting_port, count, rank):
    worker_list = []
    for i in range(count):
        worker_list.append([worker_host,
                            starting_port+i,
                            "{0}_{1}_{2}".format(suffix_id, i, rank),
                            "{0}".format(rank)
                           ])
    return worker_list

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   starting websocket server for a given client / worker node. Each now worker has a server.   #
#                                                                                               #
#***********************************************************************************************#
def start_federated_workers(worker_list, world):
    # create a process call and run all the required workers
    for i, worker in enumerate(worker_list):
        # create server command
        process_call = [PYTHON_PATH,
                        S_FILE_PATH,
                        "--host", "{0}".format(worker[0]),
                        "--port", "{0}".format(worker[1]),
                        "--id", "{0}".format(worker[2]),
                        "--rank", "{0}".format(worker[3]),
                        "--world", "{0}".format(world)]
        # run and keep track of worker processes
        PROCESS_LIST.append(subprocess.Popen(process_call))
    # start the server for new client
    print("started a total of {0} workers".format(len(worker_list)))

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   helper function to forcefully terminate all processes once Ctrl+C is hit.                   #
#                                                                                               #
#***********************************************************************************************#
def signal_handler(sig, frame):
    print("You pressed Ctrl+C!")
    for p in PROCESS_LIST:
        p.terminate()
    sys.exit(0)

#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   argument parsing and configurations for setting up the websocket server.                    #
#                                                                                               #
#***********************************************************************************************#
if __name__ == "__main__":
    
    # parsing arguments
    parser = argparse.ArgumentParser(description="Run websocket server worker.")
    parser.add_argument("--remotehost", type=str, default="localhost", help="host addres of remote server.")
    parser.add_argument("--remoteport", type=int, help="port number of federated server, e.g. --port 8778", required=True)
    parser.add_argument("--host", type=str, default="localhost", help="current and local worker's deployment host.")
    parser.add_argument("--port", type=int, help="port number current worker, e.g. --port 8778", required=True)
    parser.add_argument("--count", type=int, help="number of workers to instantiate on this machine, e.g. 5", required=True)
    parser.add_argument("--rank", type=int, help="the starting rank of workers on this machine", required=True)
    parser.add_argument("--world", type=int, help="the total number of workers in the entire federation", required=True)
    parser.add_argument("--id", type=str, help="suffix to the name (id) of the websocket workers, e.g. --id vw", required=True)
    args = parser.parse_args()
    
    # Logging setup
    FORMAT = "%(asctime)s | %(message)s"
    logging.basicConfig(format=FORMAT)
    logger = logging.getLogger("run_websocket_client")
    logger.setLevel(level=logging.DEBUG)
    
    # Websockets setup
    websockets_logger = logging.getLogger("websockets")
    websockets_logger.setLevel(logging.INFO)
    websockets_logger.addHandler(logging.StreamHandler())
    
    # create a worker list
    worker_list = generate_worker_list(args.id, args.host, args.port, args.count, args.rank)
    
    # start the required number of workers
    start_federated_workers(worker_list, args.world)
    
    # connect to server module and send detailed information
    asyncio.run(send_local_info(args.remotehost, args.remoteport, worker_list))
    
    # create a signal
    signal.signal(signal.SIGINT, signal_handler)
    signal.pause()
