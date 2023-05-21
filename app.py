from flask import Flask,request,render_template
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route("/", methods=['GET','POST'])

def predict_maths_score():
    if request.method=='GET':
        print("rendering home")
        return render_template('index.html')
    else:
        gender = request.form.get('gender')
        race = request.form.get('race')
        parent_education_lvl = request.form.get("parent_education_lvl")
        lunch_type = request.form.get("lunch_type")
        test_preparation_course = request.form.get("test_preparation_course").lower()
        reading_score = request.form.get("reading_score")
        writing_score = request.form.get("writing_score")

        custom_data = CustomData(gender,race,parent_education_lvl,lunch_type,test_preparation_course,reading_score,writing_score)

        pred_df=custom_data.make_dataframe()
        print(pred_df)

        predict_pipeline=PredictPipeline()
  
        results=predict_pipeline.predict(pred_df)
        print(f"result{results[0]}")
        return render_template('index.html',result=results[0])
        

if __name__=="__main__":
    print("Strating python flask server")
    app.run()        







