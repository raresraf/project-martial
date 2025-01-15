// Codeforces
// 110A - Nearly lucky number

#include <iostream>
#include <sstream>
#include <string>
using namespace std;

bool isLucky(string s) {
  for (unsigned int i = 0; i < s.size(); i++) {
    if (s[i] != '4' && s[i] != '7')
      return false;
  }

  return true;
}

int main() {
  string number;
  cin >> number;

  long long total = 0;

  for (unsigned int i = 0; i < number.size(); i++) {
    if (number[i] == '4' || number[i] == '7')
      total++;
  }

  stringstream ss;
  ss << total;
  string str = ss.str();

  if (isLucky(str))
    cout << "YES";
  else
    cout << "NO";

  return 0;
}
