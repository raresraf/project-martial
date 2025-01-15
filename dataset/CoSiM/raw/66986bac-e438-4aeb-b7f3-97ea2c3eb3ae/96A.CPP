#include <iostream>
#include <string>
using namespace std;
int main() {
  string input;
  cin >> input;
  string subs1 = "0000000";
  string subs2 = "1111111";
  if (input.find(subs1) != string::npos || input.find(subs2) != string::npos)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}