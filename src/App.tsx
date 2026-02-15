import { BrowserRouter, Routes, Route } from "react-router-dom";
import CameraPage from "./pages/Camera";
import Language from "./pages/Language";
import Tutorial from "./pages/Tutorial";
import Zoom from "./pages/Zoom";

function App() {
  return (
    <BrowserRouter>
      <Routes>
        <Route path="/camera" element={<CameraPage />} />
        <Route path="/" element={<Language />} />
        <Route path="/tutorial" element={<Tutorial />} />
        <Route path="/zoom" element={<Zoom />} />
      </Routes>
    </BrowserRouter>
  );
}

export default App;