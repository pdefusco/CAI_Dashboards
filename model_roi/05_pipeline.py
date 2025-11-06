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

import string
import random
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

# current date and time
now = datetime.now()

timestamp = datetime.timestamp(now)

cluster = os.getenv("CDSW_DOMAIN")
project_id = os.environ["CDSW_PROJECT_ID"]

# Set correlation factor
x = random.randint(1, 5)

# Instantiate Pipeline Util
pipelineUtil = PipelineUtil(project_id)

datagen_job, datagen_job_run = pipelineUtil.create_and_run_job(
    x=x,
    script_path="model_roi/00_datagen.py",
    job_name_prefix="datagen",
    cpu=2.0,
    memory=4.0,
    runtime_identifier="docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
    runtime_addon_identifiers=["spark351-24.1-h1"]
)

pipelineUtil.poll_job_status(
    datagen_job.id,
    datagen_job_run.id,
    poll_interval=10,
    timeout=600)

train_job, train_job_run = pipelineUtil.create_and_run_job(
    x=None,
    script_path="model_roi/01_train_model.py",
    job_name_prefix="train",
    cpu=2.0,
    memory=4.0,
    runtime_identifier="docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
    runtime_addon_identifiers=["spark351-24.1-h1"]
)

pipelineUtil.poll_job_status(
    train_job.id,
    train_job_run.id,
    poll_interval=10,
    timeout=600)

# Create the MODEL DEPLOYMENT
deploymentUtil = ModelDeployment(projectId, username)
modelName = "CLF-endpoint"

createModelResponse = deployment.createModel(projectId, modelName)
modelCreationId = createModelResponse.id

runtimeId = "docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5" #Modify as needed
createModelBuildResponse = deployment.createModelBuild(projectId, modelVersionId, modelCreationId, runtimeId)
modelBuildId = createModelBuildResponse.id

# Wait for Model Build to Complete
pipelineUtil.pipelineUtilwait_for_model_build_complete(
    model_creation_id=modelCreationId,
    model_build_id=modelBuildId,
    poll_interval=15,   # optional
    timeout=1800        # optional
)

createModelDeploymentResponse = deployment.createModelDeployment(modelBuildId, projectId, modelCreationId)

# Wait for Model Deployment to Complete
pipelineUtil.wait_for_model_deployment_status(
    model_id=modelCreationId,
    build_id=modelBuildId,
    desired_status="deployed",
    poll_interval=15,
    timeout=1800
)

simulation_job, simualtion_run = pipelineUtil.create_and_run_job(
    x=None,
    script_path="model_roi/03_simulation.py",
    job_name_prefix="simulation",
    cpu=2.0,
    memory=4.0,
    runtime_identifier="docker.repository.cloudera.com/cloudera/cdsw/ml-runtime-pbj-workbench-python3.10-standard:2025.09.1-b5",
    runtime_addon_identifiers=["spark351-24.1-h1"]
)

pipelineUtil.poll_job_status(
    simulation_job.id,
    simulation_job_run.id,
    poll_interval=10,
    timeout=600)

### Create Application

api_instance = CMLServiceApi(client)

app_response = create_cml_application(
    name="Live Model ROI Dashboard",
    subdomain="org",
    description="Continuously Updating Model ROI Dashboard",
    script="model_roi/06_live_model_roi_dashboard.py"
)
