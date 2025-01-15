#include <bits/stdc++.h>
using namespace std;

int mod(int a) { return a > 0 ? a : -a; }

int main() {
  int loc;
  int x;
  int y;

  for (int i = 0; i < 25; i++) {
    int temp;
    cin >> temp;

    if (temp == 1)
      loc = i;
  }

  x = loc / 5;
  y = (loc - 5 * x);

  // cout<<"(x,y) = ("<<x<<","<<y<<")"<<endl;

  int count = 0;

  count += mod(x - 2);
  count += mod(y - 2);

  cout << count << endl;

  return 0;
}
