#include <iostream>
#include <map>
#include <sstream>
#include <string>
using namespace std;
int main() {
  map<int, int> m1;
  m1[1] = 0;
  m1[2] = 0;
  m1[3] = 0;
  string s;
  cin >> s;
  for (int i = 0; i < s.length(); i++)
    if (s[i] != '+')
      m1[s[i] - '0'] += 1;
  string g = "";
  if (m1[1] > 0) {
    g += "1";
    m1[1] -= 1;
  } else if (m1[2] > 0) {
    g += "2";
    m1[2] -= 1;
  } else {
    g += "3";
    m1[3] -= 1;
  }

  for (int i = 1; i <= 3; i++) {
    char y = i + '0';
    string d(1, y);
    string x = "+" + (d);
    while (m1[i] > 0) {
      g = g + x;
      m1[i] -= 1;
    }
  }
  cout << g << endl;
}