/*
Queue at the School

During the break the schoolchildren, boys and girls, formed a queue of n people
in the canteen. Initially the children stood in the order they entered the
canteen. However, after a while the boys started feeling awkward for standing in
front of the girls in the queue and they started letting the girls move forward
each second.

Let's describe the process more precisely. Let's say that the positions in the
queue are sequentially numbered by integers from 1 to n, at that the person in
the position number 1 is served first. Then, if at time x a boy stands on the
i-th position and a girl stands on the (i + 1)-th position, then at time x + 1
the i-th position will have a girl and the (i + 1)-th position will have a boy.
The time is given in seconds.
*/
#include <bits/stdc++.h>
using namespace std;
int main() {
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  while (t--) {
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i++;
      }
    }
  }
  cout << s;
}