import LanguageSelector from "../components/LanguageSelector";
import { useNavigate } from "react-router-dom";
import OrthopedicImg from "../assets/orthopedic.jpg";
import { ArrowLeftIcon } from "@heroicons/react/24/outline";

export default function Tutorial() {
  const navigate = useNavigate();

  const projects = [
    {
      id: "treehacks",
      name: "Treehacks",
      image: OrthopedicImg,
    },
    {
      id: "orthopedic",
      name: "Orthopedic Drill",
      image: OrthopedicImg,
    },
        {
      id: "orthopedic",
      name: "Orthopedic Drill",
      image: OrthopedicImg,
    },
  ];

  return (
    <div className="min-h-screen flex flex-col w-full px-6 pt-10 gap-8 bg-[#F5F5F5]">
    <div className="flex items-center justify-between">
      <button
        onClick={() => navigate("/")}
        className="flex items-center gap-2 text-gray-600 hover:text-gray-900 transition"
      >
        <ArrowLeftIcon className="w-5 h-5" />
        <span className="text-sm font-medium">Back</span>
      </button>
      <div className="w-30">
        <LanguageSelector align="right" />
      </div>
    </div>
      <h2 className="text-3xl font-bold text-gray-700 text-center">
        Select Tutorial
      </h2>

      <div className="max-w-5xl mx-auto grid gap-8 sm:grid-cols-2">
        {projects.map((project) => (
          <div
            key={project.id}
            className="bg-white rounded-2xl shadow-lg overflow-hidden transition hover:shadow-xl hover:-translate-y-1"
          >
            <div className="h-56 w-full overflow-hidden">
              <img
                src={project.image}
                alt={project.name}
                className="w-full h-full object-cover"
              />
            </div>

            <div className="p-5 flex items-center justify-between border-t border-gray-200">
              <h3 className="text-xl font-semibold text-gray-900">
                {project.name}
              </h3>

              <button
                onClick={() => navigate(`/steps/${project.id}`)}
                className="bg-[#AF69EE] text-white px-4 py-2 rounded-lg text-sm font-medium transition hover:brightness-110 active:scale-95"
              >
                Start Tutorial
              </button>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
}
