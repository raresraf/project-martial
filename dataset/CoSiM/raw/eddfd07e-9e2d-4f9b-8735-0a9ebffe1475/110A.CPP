#include <iostream>
using namespace std;
int main() {
  long long int input, temp;
  cin >> input;
  int flag = 0;
  long long int count = 0;
  temp = input;
  while (temp) {
    int r = temp % 10;
    if (r == 7 || r == 4)
      count++;
    temp /= 10;
  }

  if (count == 7 || count == 4) {
    cout << "YES";
    ;
  } else
    cout << "NO";
  return 0;
}
