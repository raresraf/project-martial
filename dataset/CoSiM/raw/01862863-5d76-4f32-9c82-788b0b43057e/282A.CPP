#include <iostream>
using namespace std;
int main() {
  int n;
  string s;
  cin >> n;
  int c = 0;
  while (n--) {
    cin >> s;
    if (s == "++X" || s == "X++")
      c += 1;
    else
      c -= 1;
  }
  cout << c << endl;
}