#include <iostream>
using namespace std;
int main() {
  int n;
  cin >> n;
  cin.get();
  string s;
  getline(cin, s);
  int c = 0;
  for (int i = 1; i < n; i++) {
    if (s[i] == s[i - 1])
      c++;
  }
  cout << c;
}