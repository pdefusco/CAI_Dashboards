#****************************************************************************
# (C) Cloudera, Inc. 2020-2025
#  All rights reserved.
#
#  Applicable Open Source License: GNU Affero General Public License v3.0
#
#  NOTE: Cloudera open source products are modular software products
#  made up of hundreds of individual components, each of which was
#  individually copyrighted.  Each Cloudera open source product is a
#  collective work under U.S. Copyright Law. Your license to use the
#  collective work is as provided in your written agreement with
#  Cloudera.  Used apart from the collective work, this file is
#  licensed for your use pursuant to the open source license
#  identified above.
#
#  This code is provided to you pursuant a written agreement with
#  (i) Cloudera, Inc. or (ii) a third-party authorized to distribute
#  this code. If you do not have a written agreement with Cloudera nor
#  with an authorized and properly licensed third party, you do not
#  have any rights to access nor to use this code.
#
#  Absent a written agreement with Cloudera, Inc. (“Cloudera”) to the
#  contrary, A) CLOUDERA PROVIDES THIS CODE TO YOU WITHOUT WARRANTIES OF ANY
#  KIND; (B) CLOUDERA DISCLAIMS ANY AND ALL EXPRESS AND IMPLIED
#  WARRANTIES WITH RESPECT TO THIS CODE, INCLUDING BUT NOT LIMITED TO
#  IMPLIED WARRANTIES OF TITLE, NON-INFRINGEMENT, MERCHANTABILITY AND
#  FITNESS FOR A PARTICULAR PURPOSE; (C) CLOUDERA IS NOT LIABLE TO YOU,
#  AND WILL NOT DEFEND, INDEMNIFY, NOR HOLD YOU HARMLESS FOR ANY CLAIMS
#  ARISING FROM OR RELATED TO THE CODE; AND (D)WITH RESPECT TO YOUR EXERCISE
#  OF ANY RIGHTS GRANTED TO YOU FOR THE CODE, CLOUDERA IS NOT LIABLE FOR ANY
#  DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, PUNITIVE OR
#  CONSEQUENTIAL DAMAGES INCLUDING, BUT NOT LIMITED TO, DAMAGES
#  RELATED TO LOST REVENUE, LOST PROFITS, LOSS OF INCOME, LOSS OF
#  BUSINESS ADVANTAGE OR UNAVAILABILITY, OR LOSS OR CORRUPTION OF
#  DATA.
#
# #  Author(s): Paul de Fusco
#***************************************************************************/

"""
Utils to manage deployment of XGBoost classifier in CML
"""
from __future__ import print_function
from pprint import pprint
import cmlapi
from cmlapi.rest import ApiException
from pprint import pprint
import json, secrets, os, time

class ModelDeployment():
    """
    Class to manage the model deployment of the xgboost model
    """

    def __init__(self, projectId, username):
        self.client = cmlapi.default_client()
        self.projectId = projectId
        self.username = username

    def validatePRDProject(self, username):
        """
        Method to test successful project creation
        """

        try:
            # Return all projects, optionally filtered, sorted, and paginated.
            search_filter = {"owner.username" : username}
            search = json.dumps(search_filter)
            api_response = self.client.list_projects(search_filter=search)
            #pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->list_projects: %s\n" % e)

        return api_response

    def createModel(self, projectId, modelName, description="Enterprise AI"):
        """
        Method to create a model in Cloudera Machine Learning (CML).

        If a model with the same name already exists, creation is skipped gracefully.

        Parameters
        ----------
        self : object
            The utility class instance containing `self.client`.
        projectId : str
            The ID of the CML project in which to create the model.
        modelName : str
            The name of the model to create.
        description : str, optional
            A description of the model (default: "Enterprise AI").

        Returns
        -------
        dict or None
            The created model's API response, or None if creation was skipped.
        """
        CreateModelRequest = {
            "project_id": projectId,
            "name": modelName,
            "description": description,
            "disable_authentication": True
        }

        try:
            print(f"Creating model '{modelName}' in project {projectId}...")
            api_response = self.client.create_model(CreateModelRequest, projectId)
            pprint(api_response)
            print("Model created successfully.")
            return api_response

        except ApiException as e:
            error_message = str(e).lower()
            if "Project already has a model with that name" in error_message or "500" in error_message:
                print(f"Model '{modelName}' already exists — skipping creation.")
                return None
            else:
                print(f"Exception when calling create_model: {e}")
                return None

    def createModelBuild(self, projectId, modelCreationId, runtimeId, script_path):
        """
        Method to create a Model build
        """

        # Create Model Build
        CreateModelBuildRequest = {
                                    "runtime_identifier": runtimeId,
                                    "comment": "invoking model build",
                                    "model_id": modelCreationId,
                                    "file_path": script_path,
                                    "function_name": "predict",

                              }

        try:
            # Create a model build.
            api_response = self.client.create_model_build(CreateModelBuildRequest, projectId, modelCreationId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_build: %s\n" % e)

        return api_response

    def createModelDeployment(self, modelBuildId, projectId, modelCreationId):
        """
        Method to deploy a model build
        """

        CreateModelDeploymentRequest = {
          "cpu" : "2",
          "memory" : "4"
        }

        try:
            # Create a model deployment.
            api_response = self.client.create_model_deployment(CreateModelDeploymentRequest, projectId, modelCreationId, modelBuildId)
            pprint(api_response)
        except ApiException as e:
            print("Exception when calling CMLServiceApi->create_model_deployment: %s\n" % e)

        return api_response
