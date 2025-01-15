#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  int x = 0;
  cin.get();
  while (n--) {
    string s;
    getline(cin, s);
    if (s[1] == '+')
      x++;
    else
      x--;
  }
  cout << x;
}