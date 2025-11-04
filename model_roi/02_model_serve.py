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

import numpy
import os
import cml.metrics_v1 as metrics
import cml.models_v1 as models
from xgboost import XGBClassifier

# For JSON or text format
loaded_model = xgb.Booster()
loaded_model.load_model("my_xgboost_model.json") # or "my_xgboost_model.txt"

@models.cml_model(metrics=True)
# This is the main function used for serving the model. It will take in the JSON formatted arguments.
def explain(args):

    df = pd.DataFrame(data, index=[0])

    df['age'] = df['age'].astype(float)
    df['credit_card_balance'] = df['credit_card_balance'].astype(float)
    df['bank_account_balance'] = df['bank_account_balance'].astype(float)
    df['sec_bank_account_balance'] = df['sec_bank_account_balance'].astype(float)
    df['savings_account_balance'] = df['savings_account_balance'].astype(float)
    df['sec_savings_account_balance'] = df['sec_savings_account_balance'].astype(float)
    df['total_est_nworth'] = df['total_est_nworth'].astype(float)
    df['primary_loan_balance'] = df['primary_loan_balance'].astype(float)
    df['secondary_loan_balance'] = df['secondary_loan_balance'].astype(float)
    df['uni_loan_balance'] = df['uni_loan_balance'].astype(float)
    df['longitude'] = df['longitude'].astype(float)
    df['latitude'] = df['latitude'].astype(float)
    df['transaction_amount'] = df['transaction_amount'].astype(float)
    df['fraud_trx'] = df['fraud_trx'].astype(float)

    df.columns = ['age', 'credit_card_balance', 'bank_account_balance', 'sec_bank_account_balance', 'savings_account_balance', 'sec_savings_account_balance', 'total_est_nworth', 'primary_loan_balance', 'secondary_loan_balance', 'uni_loan_balance', 'longitude', 'latitude', 'transaction_amount', 'fraud_trx']

    y_pred = loaded_model.predict(data)[0]
    probability = loaded_model.predict_proba(data)[0]

    # Track inputs
    metrics.track_metric("input_data", data)

    # Track our prediction
    metrics.track_metric("probability", probability)

    # Track explanation
    metrics.track_metric("y_pred", y_pred)

    return {"data": dict(data), "probability": probability, "y_pred": y_pred}
