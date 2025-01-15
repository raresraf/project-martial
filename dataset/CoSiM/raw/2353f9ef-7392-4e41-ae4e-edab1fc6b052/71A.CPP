#include <iostream>
#include <string>

using namespace std;

int main() {
  int n;
  cin >> n;
  while (n--) {
    string s, output;
    cin >> s;
    if (s.length() <= 10)
      output = s;
    else {
      output = s[0] + to_string((s.length() - 2)) + s[s.length() - 1];
    }
    cout << output << endl;
  }
  return 0;
}