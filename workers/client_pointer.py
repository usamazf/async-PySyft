#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   I M P O R T     G L O B A L     L I B R A R I E S                                           #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#
import torch
import syft as sy

import binascii
from typing import Union
from typing import List

from syft.workers.websocket_client import WebsocketClientWorker
from syft.generic.abstract.tensor import AbstractTensor
from syft.messaging.message import ObjectRequestMessage

import websockets

#-----------------------------------------------------------------------------------------------#
#                                                                                               #
#   Define global parameters.                                                                   #
#                                                                                               #
#-----------------------------------------------------------------------------------------------#


#***********************************************************************************************#
#                                                                                               #
#   description:                                                                                #
#   class that implements the logic for Federated Server corresponding to worker nodes.         #
#                                                                                               #
#***********************************************************************************************#
class FederatedWorkerPointer(WebsocketClientWorker):
    def __init__(
        self,
        hook,
        host: str,
        port: int,
        secure: bool = False,
        id: Union[int, str] = 0,
        is_client_worker: bool = False,
        log_msgs: bool = False,
        verbose: bool = False,
        data: List[Union[torch.Tensor, AbstractTensor]] = None,
        timeout: int = None,
    ):
        """A client which will forward all messages to a remote worker running a
        WebsocketServerWorker and receive all responses back from the server.
        """

        # call WebsocketClientWorker constructor
        super().__init__(
            hook=hook,
            host=host,
            port=port,
            secure=secure,
            id=id,
            is_client_worker=is_client_worker,
            log_msgs=log_msgs,
            verbose=verbose,
            data=data,
            timeout=timeout,
        )

    async def set_train_config(self, **kwargs):
        """Call the set_train_config() method on the remote worker (FederatedWorker instance).
        Args:
            **kwargs:
                return_ids: List[str]
        """
        # send the training configuration and get response
        return_ids = kwargs["return_ids"] if "return_ids" in kwargs else [sy.ID_PROVIDER.pop()]
        response = self._send_msg_and_deserialize("set_train_config", return_ids=return_ids, **kwargs)
        # return the reponse from the above call.
        return response

    async def async_fit(self, dataset_key: str, epoch: int, device: str = "cpu", return_ids: List[int] = None):
        """Asynchronous call to fit function on the remote location.
        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            return_ids: List of return ids.
        Returns:
            See return value of the FederatedWorker.fit() method.
        """
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(self.url, timeout=self.timeout, max_size=None, ping_timeout=self.timeout) as websocket:
            message = self.create_worker_command_message(
                command_name="fit", return_ids=return_ids, dataset_key=dataset_key, epoch=epoch, device=device
            )
            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)
            await websocket.send(str(binascii.hexlify(serialized_message)))
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()
        
        print("\n\n\n\n\nwe were definitely here")
        print(self.list_objects_remote())
        print("\n\n\n\n\n")
        
        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage(return_ids[0], None, "")
        serialized_message = sy.serde.serialize(msg)
        loss = self._send_msg(serialized_message)
        
        msg = ObjectRequestMessage(return_ids[1], None, "")
        serialized_message = sy.serde.serialize(msg)
        updated_params = self._send_msg(serialized_message)
        
        # Return the deserialized response.
        return sy.serde.deserialize(loss), sy.serde.deserialize(updated_params)

    async def async_fit_add_sec_share(self, dataset_key: str, epoch: int, device: str = "cpu", return_ids: List[int] = None):
        """Asynchronous call to fit function on the remote location.
        Args:
            dataset_key: Identifier of the dataset which shall be used for the training.
            return_ids: List of return ids.
        Returns:
            See return value of the FederatedWorker.fit() method.
        """
        if return_ids is None:
            return_ids = [sy.ID_PROVIDER.pop()]

        # Close the existing websocket connection in order to open a asynchronous connection
        # This code is not tested with secure connections (wss protocol).
        self.close()
        async with websockets.connect(self.url, timeout=self.timeout, max_size=None, ping_timeout=self.timeout) as websocket:
            message = self.create_worker_command_message(
                command_name="fit", return_ids=return_ids, dataset_key=dataset_key, epoch=epoch, device=device
            )
            # Send the message and return the deserialized response.
            serialized_message = sy.serde.serialize(message)
            await websocket.send(str(binascii.hexlify(serialized_message)))
            await websocket.recv()  # returned value will be None, so don't care

        # Reopen the standard connection
        self.connect()

        # Send an object request message to retrieve the result tensor of the fit() method
        msg = ObjectRequestMessage(return_ids[0], None, "")
        serialized_message = sy.serde.serialize(msg)
        updated_chunk = self._send_msg(serialized_message)
        
        loss = None
        
        # Return the deserialized response.
        return loss, sy.serde.deserialize(updated_chunk)

