import LanguageSelector from "../components/LanguageSelector";
import { useNavigate } from "react-router-dom";

import { ArrowRightIcon } from "@heroicons/react/24/outline";

export default function Language(){
  const navigate = useNavigate();
  return (
    <div className="min-h-screen flex flex-col justify-center w-full px-6 pt-10 gap-8">
      <h2 className="text-3xl font-bold text-gray-700 text-center">
        Select language
      </h2>
      <div className="w-full max-w-sm mx-auto flex flex-col gap-4">
        <LanguageSelector align="center"/>
        <div className="flex justify-end">
          <button className="inline-flex items-center gap-2 bg-[#AF69EE] text-white px-5 py-3 rounded-xl transition active:scale-95"
          onClick={() => navigate("/tutorial")}>
            <ArrowRightIcon className="w-4 h-4" />
          </button>
        </div>
      </div>
      <button></button>
    </div>
  );
}