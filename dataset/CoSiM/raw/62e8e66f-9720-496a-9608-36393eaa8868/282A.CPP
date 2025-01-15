#include <iostream>
#include <string>

using namespace std;

int main() {
  int t;
  cin >> t;
  int count = 0;
  for (int i = 0; i < t; i++) {
    string s;
    cin >> s;
    if (s.at(0) == '+' || s.at(s.length() - 1) == '+') {
      count++;
    } else {
      count--;
    }
  }
  cout << count << endl;
}