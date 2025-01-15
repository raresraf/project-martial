#include <iostream>
#include <string>

using namespace std;

int main() {
  string s;
  int n, x = 0;
  cin >> n;
  for (int i = 0; i < n; i++) {
    cin >> s;
    if (s == "X++" || s == "++X") {
      x++;
    } else if (s == "X--" || s == "--X") {
      x--;
    }
  }
  cout << x << endl;
  return 0;
}