#include <bits/stdc++.h>
using namespace std;
typedef long long ll;

int main() {
  string s;
  bool test = true;
  cin >> s;
  if (s.find('h') != string::npos) {
    s = s.substr(s.find('h') + 1);
  } else {
    test = false;
  }
  if (s.find('e') != string::npos) {
    s = s.substr(s.find('e') + 1);
  } else {
    test = false;
  }
  if (s.find('l') != string::npos) {
    s = s.substr(s.find('l') + 1);
  } else {
    test = false;
  }
  if (s.find('l') != string::npos) {
    s = s.substr(s.find('l') + 1);
  } else {
    test = false;
  }
  if (s.find('o') != string::npos) {
    s = s.substr(s.find('o') + 1);
  } else {
    test = false;
  }
  if (test) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}