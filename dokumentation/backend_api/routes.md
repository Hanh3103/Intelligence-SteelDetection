# Rest endpoints

- [Registrate a user](#user-registrate)
- [Log in a user](#user-login)
- [Get current user](#get-current-user)
- [Upload Excel/CSV file](#upload-a-file)
- [Get all uploaded files](#get-all-uploaded-files)
- [Delete a file on server](#delete-a-file-on-server)
- [Make prediction with supervised models](#make-prediction-with-supervised-models)
- [Make prediction with unsupervised models](#make-prediction-with-unsupervised-models)

## User registrate
### Model

<code>id: int?, username: string, fullname:string, email:string, password: string, disabled: boolean</code>

Beispiel:

````
{
    "id": "1",
    "username": "exampleUsername",
		"fullname":"full name of user",
		"email": "test@mail.de",
    "password": "$2a$10$9X.BJxMFiHvYXksZvX3TheqzO4SPuFpG/LIYc.nkdx2eS.UhGfgvy",
		"disabled": "false"
}
````

### Routes

User resgistrate:
<pre>POST /registrate
Request Body: User Object as JSON without ID
Response: New created user
Success: 200
Error: 400, 500
</pre>

## User login
With user login who can access to the functions of dashboard. Every user is currently treated as an admin. With the returned token, for example, 
User & Prediction requests can be carried out. 

### Route
User log in:
<pre>POST /token
Request: User with ID ({ username: "USERNAME", password: "PASSWORD" })
Response: { access_token: "TOKEN", token_type: "Bearer" }
Success: 200
Error: 400, 401 (Incorrect username or password), 500
</pre>

## Get current user
With the access token from /token, the information of current user will be retured.  

### Route
Get current user:
<pre>GET /users/me
Authentication: Bearer-Token from the logged-in user
Response: user object as Json without ID and password
Success: 200
Error: 401(Could not validate credentials)
</pre>

## Upload a file
### Model
<code>id: int?, owner_id: integer, filename: string, filesize: integer, namedb: string</code>
Beispiel:

````
{
    "id": "1",
    "filename": "test.csv",
		"filesize":"1497",
		"namedb": "hanh3_test.csv"
}
````

### Route

<pre>POST /user/me/item
Authentication: Bearer-Token from the logged-in user
Request: content, filename and filesize of the uploaded file as Json 
Response: Item uploaded successfully 
Success: 200
Error: 403, 400
</pre>

## Get all uploaded files
with the token return all the uploaded files of the current logging in user. The names of all files are the saved name of the files in database 
(s. column "namedb") 

### Route
<pre>GET /user/me/items
Authentication: Bearer-Token from the logged-in user
Response: name of the all files and their file contents as Json  
Success: 200
Error: 404 (No items found)
</pre>

## Delete a file on server
Delete a file with the name of this file in database. 

### Route
<pre>GET /user/me/item/{table_name}
Authentication: Bearer-Token from the logged-in user
Request: name of the file
Response: {"message": "Item deleted successfully"}
Success: 200
Error: 404 (No items found), 400 (Item could not be deleted successfully due to an error in the database)
</pre>

## Make prediction with supervised models
There are 2 mode of configuration for prediction:
- prediction with existing model: the user can upload their model and encoder then make prediction with theses
- prediction with trained model: the user can upload their encoder and then trained the model with uploaded file and then make prediction 
- There is 2 algorithms to choose

### Route
<pre>GET /predict/supervised
Authentication: Bearer-Token from the logged-in user
Request: files with payload of configuration as Json
Response: results of prediction
Success: 200
Error: 400, 404
</pre>

## Make prediction with unsupervised models
### Route
Kmeans
<pre>POST /predict/unsupervised/kmeans
Request: Dictionary containing model options and DataFrame for prediction.
Response: JSON representation of the clustered data.
Success: 200
Error: 
</pre>

GMM
<pre>POST /predict/unsupervised/gmm
Request: Dictionary containing model options and DataFrame for prediction.
Response: JSON representation of the clustered data.
Success: 200
Error: 
</pre>

Hierarchical
<pre>POST /predict/unsupervised/hierarchical
Request: Dictionary containing model options and DataFrame for prediction.
Response: JSON representation of the clustered data.
Success: 200
Error: 
</pre>

Tsne: t-Distributed Stochastic Neighbor Embedding
<pre>POST /predict/unsupervised/tsne
Request: Dictionary containing model options and DataFrame for prediction.
Response: JSON representation of the clustered data.
Success: 200
Error: 
</pre>

Pca: Principal Component Analysis (PCA).
<pre>POST /predict/unsupervised/pca
Request: Dictionary containing model options and DataFrame for prediction.
Response: JSON representation of the clustered data.
Success: 200
Error: 
</pre>

Umap: Uniform Manifold Approximation and Projection (UMAP)
<pre>POST /predict/unsupervised/umap
Request: Dictionary containing model options and DataFrame for prediction.
Response: JSON representation of the clustered data.
Success: 200
Error: 
</pre>








