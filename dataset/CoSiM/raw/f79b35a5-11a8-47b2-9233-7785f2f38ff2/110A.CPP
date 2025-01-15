// http://codeforces.com/problemset/problem/110/A

#include <iostream>
#include <string>
using namespace std;

int main() {
  long long n;
  cin >> n;
  int number = 0;
  while (n > 0) {
    if (n % 10 == 4 or n % 10 == 7) {
      number++;
    }
    n /= 10;
  }
  if (number == 0) {
    cout << "NO";
    return 0;
  }
  int flag = 0;
  while (number > 0) {
    if (number % 10 != 4 and number % 10 != 7) {
      flag++;
    }
    number /= 10;
  }
  if (flag == 0) {
    cout << "YES";
  } else
    cout << "NO";

  return 0;
}