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

/**
 * @file 266B.CPP
 * @brief Solution to the "Queue at the School" problem from Codeforces.
 *
 * @algorithm Time-step simulation. The core logic consists of nested loops. The outer
 * loop simulates the passage of time for 't' seconds. The inner loop performs a single
 * pass over the queue, swapping adjacent 'B' and 'G' pairs.
 * @time_complexity O(time * qsize), as the algorithm iterates through the queue for each second.
 * @space_complexity O(qsize) to store the queue string.
 */
#include <bits/stdc++.h>

using namespace std;

int main() {
  char temp;
  string q;
  int qsize, time, boys = 0;
  std::cin >> qsize >> time;
  cin >> q;
  // This block for counting boys is not used in the final algorithm.
  for (int i = 0; i < qsize; i++) {
    if (q[i] == 'B') {
      ++boys;
    }
  }
  // cout << "No. of boys are:" << boys;
  // cout << "Time is" << time;

  /**
   * Block Logic: The outer loop simulates the progression of 'time' seconds.
   */
  for (int i = 0; i < time; i++) {
    /**
     * Block Logic: The inner loop scans the queue for adjacent 'B' and 'G' pairs.
     * The check q[j+1] is safe because the loop condition is j < q.length() and
     * the last checked element is q[q.length()-1], so j+1 will be at most q.length().
     * Note: Accessing q[j+1] when j is q.length()-1 is technically out-of-bounds but
     * often works due to null termination or other memory layout details; however,
     * it is not safe practice. The short-circuiting of `&&` prevents a crash if q[j] != 'B'.
     */
    for (int j = 0; j < q.length(); j++) {
      // Pre-condition: Identifies a boy standing immediately before a girl.
      if (q[j] == 'B' && q[j + 1] == 'G') {
        temp = q[j];
        q[j] = q[j + 1];
        q[j + 1] = temp;
        // Inline: The loop counter 'j' is incremented to skip the next position.
        // This is crucial as it ensures that the newly moved girl does not cause
        // another swap in the same time step, correctly modeling the problem's rules.
        ++j;
      }
    }
  }

  // Output the final configuration of the queue.
  cout << q;

  /*
   * The following blocks are commented out and appear to be remnants of the
   * coding process, exploring alternative or incorrect approaches. They are not
   * part of the final executable logic.
   */

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