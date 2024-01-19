import os

from fastapi import APIRouter
from fastapi import Depends, HTTPException, status, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import Json
from sqlalchemy.orm import Session

from db.db import (get_current_active_user, get_db, get_dataframe_from_database)
from db.dbModels import UserInDB
from utils.supervised_utils import parse_payload, extract_modul_class_name, existing_model_classifier, \
    prediction_with_algorithm, train_model_then_predict_classifier, save_trained_model_co_on_server

router = APIRouter()


@router.post("/predict/supervised")
async def predict_supervised(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        upload_model_file: UploadFile = File(...),
        encoded_file: UploadFile = File(...),
        up_model_options: Json = Form(...)
):
    # parse the payload
    try:
        algorithm, existing_model, table_name_to_predict, table_name_to_train, name_of_function = await parse_payload(
            up_model_options)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="parse payload problem" + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # get the actual dataframe from the database
    try:
        table_to_predict_df = get_dataframe_from_database(table_name_to_predict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Table to predict not found in database " + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    module_name, class_name = await extract_modul_class_name(algorithm)

    if existing_model == "True":
        classifier = await existing_model_classifier(class_name, encoded_file, module_name, parent_dir,
                                                     upload_model_file)
        # only prediction works !!!
        try:
            results = prediction_with_algorithm(classifier, table_to_predict_df, algorithm)
            return {"prediction": results}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="prediction existing model has problems" + str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        classifier = await train_model_then_predict_classifier(class_name, encoded_file, module_name, parent_dir,
                                                               table_name_to_train)
        # predictions + save the model + test the model
        try:
            results = prediction_with_algorithm(classifier, table_to_predict_df, algorithm)  # list
            test_results = classifier.test_model()  # string
            name_new_model, name_new_encoder, encoded_model, encoded_encoder = await save_trained_model_co_on_server(
                algorithm, classifier, parent_dir)
            json_response = JSONResponse(content={"prediction": results,
                                                  "test_results": test_results,
                                                  "name_new_model": name_new_model,
                                                  "name_new_encoder": name_new_encoder,
                                                  "encoded_model": encoded_model,
                                                  "encoded_encoder": encoded_encoder})
            return json_response
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="prediction and saving have problems " + str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
