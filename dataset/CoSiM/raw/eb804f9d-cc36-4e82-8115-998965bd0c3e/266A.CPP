#include <iostream>
#include <string>

using namespace std;

int main() {
  int count = 0;
  int t;
  cin >> t;
  string s;
  cin >> s;
  for (int i = 0; i < s.length() - 1; i++) {
    if (s[i] == s[i + 1]) {
      count++;
    }
  }
  cout << count << endl;
}