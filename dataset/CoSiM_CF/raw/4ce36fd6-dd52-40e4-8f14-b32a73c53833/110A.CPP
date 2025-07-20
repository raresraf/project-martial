#include <iostream>
#include <string.h>
using namespace std;
int check_lucky(int nlucky);
int main() {
  string number;
  cin >> number;
  int numlength(number.length()), nlucky(0), iter(0);
  for (iter = 0; iter < numlength; iter++) {
    if (number[iter] == '4' || number[iter] == '7') {
      nlucky++;
    } else {
      continue;
    }
  }
  if (check_lucky(nlucky) == 1) {
    cout << "YES" << endl;
  } else {
    cout << "NO" << endl;
  }
  return 0;
}

int check_lucky(int nlucky) {
  string nluckyString = to_string(nlucky);
  int length(nluckyString.length()), iter(0);
  for (iter = 0; iter < length; iter++) {
    if (nluckyString[iter] == '4' || nluckyString[iter] == '7') {
      continue;
    } else {
      return 0;
    }
  }
  return 1;
}