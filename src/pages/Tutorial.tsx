import LanguageSelector from "../components/LanguageSelector";
export default function Tutorial(){
  return (
    <div className="min-h-screen flex flex-col justify-center w-full px-6 pt-10 gap-8">
      <div className="flex justify-end">
        <div className="justify-end w-30">
          <LanguageSelector align="right"/>
        </div>
      </div>
      <h2 className="text-3xl font-bold text-gray-700 text-center">
        Select tutorial
      </h2>
    </div>
  );
}