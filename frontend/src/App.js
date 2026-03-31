import { useState } from "react";

import Header from "./Component/Header";
import Insert from "./Component/Insert";
import Chat from "./Component/Chat";
import Preview from "./Component/Preview";
import Loading from "./Component/Loading";
import Prediction from "./Component/Prediction";
import Reupload from "./Component/Reupload";

const App = () => {
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false); 
  const [id, setID] = useState(null);
  const BASE_URL = process.env.REACT_APP_API_URL;

  return (
    <div className="min-h-full h-auto">
      <header className="flex">
        <div className="">
          <Header />
        </div>
      </header>

      <main className="flex flex-col items-center">
        <div>
          {loading && <Loading />}
          {id && <Prediction id={id} setLoading={setLoading} url={BASE_URL} />}
          {!id && !loading && (
            file ? <Preview file={file} setFile={setFile} /> : <Insert setFile={setFile} />
          )}
        </div>

        {!id && (
          <div className="flex flex-col">
            <div>
              <Chat file={file} setFile={setFile} setLoading={setLoading} 
              id={id} setID={setID} url={BASE_URL}/>
            </div>
          </div>
        )}
        {id && (
          <Reupload setID={setID}/>
        )

        }
      </main>
    </div>
  );
}

export default App;
