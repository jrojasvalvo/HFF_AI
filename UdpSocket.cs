using UnityEngine;
using System.Collections;
using System;
using System.Text;
using System.Net;
using System.Net.Sockets;
using System.Threading;

// Borrowed from https://github.com/Siliconifier/Python-Unity-Socket-Communication except SendDataCoroutine has been
// heavily modified for my purposes
public class UdpSocket : MonoBehaviour
{
    [HideInInspector] public bool isTxStarted = false;

    [SerializeField] string IP = "127.0.0.1"; // local host
    [SerializeField] int rxPort = 8000; // port to receive data from Python on
    [SerializeField] int txPort = 8001; // port to send data to Python on

    // Create necessary UdpClient objects
    UdpClient client;
    IPEndPoint remoteEndPoint;
    Thread receiveThread; // Receiving Thread

    private Vector3 playerPos;
    private double player_x;
    private double player_y;
    private double player_z;

    private Vector3 finishPos;
    private double finish_x;
    private double finish_y;
    private double finish_z;

    private double distance;

    private double start_x;
    private double start_z;
    private double from_start;

    IEnumerator SendDataCoroutine() 
    {
        while (true)
        {
            print("SENDING");
            playerPos = GameObject.FindGameObjectsWithTag("Player")[1].transform.position;
            player_x = playerPos.x;
            player_y = playerPos.y;
            player_z = playerPos.z;

            start_x = 0.0;
            start_z = 0.0;

            finishPos = GameObject.FindGameObjectsWithTag("Finish")[0].transform.position;
            finish_x = finishPos.x;
            finish_z = finishPos.z;

            distance = Mathf.Sqrt((float)((finish_x - player_x) * (finish_x - player_x) + (finish_z - player_z) * (finish_z - player_z)));
            from_start = Mathf.Sqrt((float)(player_x * player_x + player_z * player_z));

            if(player_y > -5.0)
            {SendData(start_x.ToString("N1") + " " + start_z.ToString("N1") + " " + player_x.ToString("N1") + " " + player_z.ToString("N1") + " " + finish_x.ToString("N1") + " " + finish_z.ToString("N1") + " " + distance.ToString("N1") + " " + from_start.ToString("N1"));}
            else{SendData("DEAD");}
            
            yield return new WaitForSeconds(1f);
        }
    }

    
    public void SendData(string message) // Use to send data to Python
    {
        try
        {
            byte[] data = Encoding.UTF8.GetBytes(message);
            client.Send(data, data.Length, remoteEndPoint);
        }
        catch (Exception err)
        {
            print(err.ToString());
        }
    }

    void Awake()
    {
        // Create remote endpoint (to Matlab) 
        remoteEndPoint = new IPEndPoint(IPAddress.Parse(IP), txPort);

        // Create local client
        client = new UdpClient(rxPort);

        StartCoroutine(SendDataCoroutine()); // DELETE THIS: Added to show sending data from Unity to Python via UDP
    }

    //Prevent crashes - close clients and threads properly!
    void OnDisable()
    {
        client.Close();
    }

}