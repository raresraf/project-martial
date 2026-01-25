// 282A
#include <cstring>
#include <iostream>
using namespace std;
int main() {
  int n;
  string s;
  int res = 0;
  cin >> n;
  while (n--) {
    cin >> s;
    if (s == "X++" || s == "++X")
      res++;
    else if (s == "X--" || s == "--X")
      res--;
  }
  cout << res;
  return 0;
}
