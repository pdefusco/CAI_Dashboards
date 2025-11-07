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

from __future__ import print_function
import cmlapi
from cmlapi.rest import ApiException
from datetime import datetime
from pprint import pprint
import json, secrets, os, time
import mlflow
from mlops import ModelDeployment
from cmlapi.utils import Cursor
from mlops import ModelDeployment
from pipelineUtils import PipelineUtil
import random
import string

# current date and time
now = datetime.now()

timestamp = datetime.timestamp(now)

username = os.environ["PROJECT_OWNER"]
cluster = os.getenv("CDSW_DOMAIN")
project_id = os.environ["CDSW_PROJECT_ID"]

# Set correlation factor
x = random.randint(1, 5)

# Instantiate Pipeline Util
pipelineUtil = PipelineUtil(project_id, username)

def random_suffix(length=5):
    """Generate a random alphanumeric string of given length."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

for i in range(10):
    print(f"\nRunning pipeline iteration {i+1}/10\n")

    # Create and run datagen job
    datagen_job_run_response = pipelineUtil.create_and_run_job(
        x=x,
        script_path="model_roi/00_datagen.py",
        job_name_prefix="datagen",
        cpu=2.0,
        memory=4.0,
        runtime_identifier="docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
        runtime_addon_identifiers=["spark351-24.1-h1"]
    )

    pipelineUtil.poll_job_status(
        datagen_job_run_response.job_id,
        datagen_job_run_response.id,
        poll_interval=10,
        timeout=600
    )

    # Create and run train job
    train_job_run_response = pipelineUtil.create_and_run_job(
        x=None,
        script_path="model_roi/01_train_model.py",
        job_name_prefix="train",
        cpu=2.0,
        memory=4.0,
        runtime_identifier="docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
        runtime_addon_identifiers=["spark351-24.1-h1"]
    )

    pipelineUtil.poll_job_status(
        train_job_run_response.job_id,
        train_job_run_response.id,
        poll_interval=10,
        timeout=600
    )

    # Create unique model name
    unique_model_name = f"CLF-endpoint-{random_suffix()}"

    # Create the model
    deployment = ModelDeployment(project_id, username)
    createModelResponse = deployment.createModel(project_id, unique_model_name)

    if createModelResponse is None:
        print(f"Model '{unique_model_name}' already exists — skipping build/deployment.")
        continue  # Skip this iteration if model already exists

    modelCreationId = createModelResponse.id
    runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5"

    # Create model build
    createModelBuildResponse = deployment.createModelBuild(project_id, modelCreationId, runtimeId, script_path="model_roi/02_model_serve.py")
    modelBuildId = createModelBuildResponse.id

    # Wait for Model Build to Complete
    pipelineUtil.wait_for_model_build_complete(
        model_creation_id=modelCreationId,
        model_build_id=modelBuildId,
        desired_status="built",
        poll_interval=15,
        timeout=1800
    )

    # Create model deployment
    createModelDeploymentResponse = deployment.createModelDeployment(modelBuildId, project_id, modelCreationId)

    # Wait for Model Deployment to Complete
    pipelineUtil.wait_for_model_deployment_status(
        model_id=modelCreationId,
        build_id=modelBuildId,
        desired_status="deployed",
        poll_interval=15,
        timeout=1800
    )

    # Run simulation job
    simulation_job_response = pipelineUtil.create_and_run_job(
        x=unique_model_name,
        script_path="model_roi/03_simulation.py",
        job_name_prefix="simulation",
        cpu=2.0,
        memory=4.0,
        runtime_identifier="docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
        runtime_addon_identifiers=["spark351-24.1-h1"]
    )

    pipelineUtil.poll_job_status(
        simulation_job_response.job_id,
        simulation_job_response.id,
        poll_interval=10,
        timeout=600
    )

    print(f"\nIteration {i+1} completed.\n")

### Create Application

api_instance = CMLServiceApi(client)

app_response = create_cml_application(
    name="Live Model ROI Dashboard",
    subdomain="org",
    description="Continuously Updating Model ROI Dashboard",
    script="model_roi/06_live_model_roi_dashboard.py"
)




# Create unique model name
unique_model_name = f"CLF-endpoint-{random_suffix()}"

# Create the model
deployment = ModelDeployment(project_id, username)
createModelResponse = deployment.createModel(project_id, unique_model_name)

if createModelResponse is None:
    print(f"Model '{unique_model_name}' already exists — skipping build/deployment.")

modelCreationId = createModelResponse.id
runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5"
desired_status='built'
# Create model build
createModelBuildResponse = deployment.createModelBuild(project_id, modelCreationId, runtimeId, script_path="model_roi/01_train_model.py")
modelBuildId = createModelBuildResponse.id

client = cmlapi.default_client()
while True:
    try:
        # Query model deployment status
        api_response = client.list_model_builds(
            project_id, modelCreationId,
        )
        print("\nModel Build Status: ")
        pprint(api_response.model_builds[0].status)

        # If build is complete, break out
        if api_response.model_builds[0].status.lower() == desired_status.lower():
            print("Model build has reached 'built' state.")

        # Wait before polling again

    except Exception as e:
        print(f"Exception when calling CMLServiceApi->list_model_deployments: {e}")



createModelDeploymentResponse = deployment.createModelDeployment(modelBuildId, project_id, modelCreationId)
api_response = client.list_model_deployments(
                    project_id,
                    modelCreationId,
                    modelBuildId
                )
pprint(api_response)
