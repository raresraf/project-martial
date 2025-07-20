/*During the break the schoolchildren, boys and girls, formed a queue of n
people in the canteen. Initially the children stood in the order they entered
the canteen. However, after a while the boys started feeling awkward for
standing in front of the girls in the queue and they started letting the girls
move forward each second.

Let's describe the process more precisely. Let's say that the positions in the
queue are sequentially numbered by integers from 1 to n, at that the person in
the position number 1 is served first. Then, if at time x a boy stands on the
i-th position and a girl stands on the (i + 1)-th position, then at time x + 1
the i-th position will have a girl and the (i + 1)-th position will have a boy.
The time is given in seconds.

You've got the initial position of the children, at the initial moment of time.
Determine the way the queue is going to look after t seconds.

Input
The first line contains two integers n and t (1   n, t   50), which represent
the number of children in the queue and the time after which the queue will
transform into the arrangement you need to find.

The next line contains string s, which represents the schoolchildren's initial
arrangement. If the i-th position in the queue contains a boy, then the i-th
character of string s equals "B", otherwise the i-th character equals "G".

Output
Print string a, which describes the arrangement after t seconds. If the i-th
position has a boy after the needed time, then the i-th character a must equal
"B", otherwise it must equal "G".*/

#include <bits/stdc++.h>

using namespace std;

int main() {
  char temp;
  string q;
  int qsize, time, boys = 0;
  std::cin >> qsize >> time;
  cin >> q;
  for (int i = 0; i < qsize; i++) {
    if (q[i] == 'B') {
      ++boys;
    }
  }
  // cout << "No. of boys are:" << boys;
  // cout << "Time is" << time;

  for (int i = 0; i < time; i++) {
    for (int j = 0; j < q.length(); j++) {
      if (q[j] == 'B' && q[j + 1] == 'G') {
        temp = q[j];
        q[j] = q[j + 1];
        q[j + 1] = temp;
        ++j;
      }
    }
  }

  cout << q;

  /*while(time--)
  {
      for(int i = 0; i < q.length(); i++)
      {
          if(q)
      }
  }
  */
  return 0;
}

/*int countFreq(string &pat, string &txt)
{
    int M = pat.length();
    int N = txt.length();
    int res = 0;

    /* A loop to slide pat[] one by one */
/*    for (int i = 0; i <= N - M; i++)
    {
        /* For current index i, check for
           pattern match */
/*        int j;
        for (j = 0; j < M; j++)
            if (txt[i+j] != pat[j])
                break;

        // if pat[0...M-1] = txt[i, i+1, ...i+M-1]
        if (j == M)
        {
           res++;
           j = 0;
        }
    }
    return res;
} */

/*for (int i = 0; i < boys; i++)
    {
        while(time--)
        {
            if(q[i] == 'B' && q[i+1] == 'G')
            {
                temp = q[i];
                q[i] = q[i+1];
                q[i+1] = temp;
            }
        }
        //cout << "Hi";
    }
*/
