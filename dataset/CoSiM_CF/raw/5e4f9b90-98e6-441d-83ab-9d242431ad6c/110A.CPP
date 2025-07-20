#include <iostream>
#include <sstream>
#include <string>

using namespace std;

int main() {
  string number;
  cin >> number;

  int i = 0, j = 0;
  ;
  bool isLucky;
  while (number[i] != '\0') {
    if (number[i] == '4' || number[i] == '7')
      j++;
    i++;
  }
  string digit;
  ostringstream convert;
  convert << j++;
  digit = convert.str();
  i = 0;
  bool digitLucky = true;
  while (digit[i] != '\0') {
    if (digit[i] != '4' && digit[i] != '7') {
      digitLucky = false;
      break;
    }
    i++;
  }
  if (digitLucky)
    cout << "YES";
  else
    cout << "NO";
  return 0;
}