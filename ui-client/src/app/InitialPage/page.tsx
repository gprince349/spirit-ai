import Image from "next/image";

export const InitialPage = () => {

  return (
    <div className="  h-screen bg-[url('/first_screen.png')] bg-cover bg-center bg-no-repeat ">
      <div className="flex flex-col justify-between  p-8 text-slate-800 text-center relative">
        <Image src="/spiritual_lotus.png" alt="Spiritual AI Logo" width={100} height={100} 
        className="mx-auto flex flex-col" />
        <h1 className="text-5xl font-bold">Spiritual AI</h1>
        <p className=" mt-4 w-[25%] mx-auto text-2xl font-medium" >Your companion for inner peace and clearity </p>
      </div>

      {/* <div>
        {/* progress of logging in *}
        <progress value="0" max="100"></progress>
      </div> */}
    </div>
  )
}