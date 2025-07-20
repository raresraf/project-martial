// Author: Cheng(Steve) Hang
// Contact: calvinhance@gmail.com
// All  rights reserved.
// Site: njuapp.github.io
// Date: 2018-02-11 20:25:35
// Last modified: 2018-02-12 17:11:32
#include <iostream>
#include <string>
using namespace std;
int main() {
  int num;
  cin >> num;
  int x = 0;
  while (num--) {
    string s;
    cin >> s;
    if (s == "++X" || s == "X++")
      x++;
    else
      x--;
  }
  cout << x << endl;
  return 0;
}
