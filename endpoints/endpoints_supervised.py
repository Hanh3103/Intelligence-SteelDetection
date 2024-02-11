import os

from fastapi import APIRouter
from fastapi import Depends, HTTPException, status, File, UploadFile, Form
from fastapi.responses import JSONResponse
from pydantic import Json
from sqlalchemy.orm import Session

from db.db_instance import get_current_active_db_session as get_db
from utils.database_utils import get_current_active_user, get_dataframe_from_database
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
    """
    Endpoint for predicting with a supervised algorithm.

    Args:
        db (Session): Database session dependency.
        current_user (UserInDB): Current authenticated user dependency.
        upload_model_file (UploadFile): Uploaded model file.
        encoded_file (UploadFile): Uploaded encoded file.
        up_model_options (Json): JSON payload containing algorithm details.

    Returns:
        JSONResponse: Response containing prediction results, test results, and model details.

    Raises:
        HTTPException: Raised for various HTTP status codes with corresponding details.

    Notes:
        - Parses the payload to extract algorithm details.
        - Retrieves the DataFrame for prediction from the database.
        - Determines whether to use an existing model or train a new one.
        - Performs prediction, testing, and optionally saves the new model on the server.
    """
    # Parse the payload
    try:
        algorithm, existing_model, table_name_to_predict, table_name_to_train, name_of_function = await parse_payload(
            up_model_options)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Parse payload problem: " + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Get the actual DataFrame from the database
    try:
        table_to_predict_df = get_dataframe_from_database(table_name_to_predict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Table to predict not found in database: " + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Determine module and class names from the algorithm
    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    module_name, class_name = await extract_modul_class_name(algorithm)

    # Handle existing or new model scenarios
    if existing_model == "True":
        # Use an existing model for prediction
        try:
            classifier = await existing_model_classifier(class_name, encoded_file, module_name, parent_dir,
                                                         upload_model_file)
            results = prediction_with_algorithm(classifier, table_to_predict_df, algorithm)
            return {"prediction": results}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Prediction with existing model has problems: " + str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        # Train a new model, perform predictions, and potentially save the new model
        try:
            classifier = await train_model_then_predict_classifier(class_name, encoded_file, module_name, parent_dir,
                                                                   table_name_to_train)
            results = prediction_with_algorithm(classifier, table_to_predict_df, algorithm)
            test_results = classifier.test_model()
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
                detail="Prediction and saving have problems: " + str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
