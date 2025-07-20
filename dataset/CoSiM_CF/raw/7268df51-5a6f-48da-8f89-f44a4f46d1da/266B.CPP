#include <iostream>
#include <set>
using namespace std;
int main() {
  // your code goes here
  int n, t;
  cin >> n >> t;
  string s;
  cin >> s;
  for (int i = 0; i < t; i++) {
    for (int i = 0; i < n - 1; i++) {
      if (s[i] == 'B' && s[i + 1] == 'G') {
        s[i] = 'G';
        s[i + 1] = 'B';
        i = i + 1;
      }
    }
  }
  cout << s;
  return 0;
}