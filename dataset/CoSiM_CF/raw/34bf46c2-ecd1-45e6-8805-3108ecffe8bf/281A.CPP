#include <algorithm>
#include <iostream>
#include <vector>
using namespace std;
int year, i;
int main() {
  cin >> year;
  while (true) {
    year = year + 1;
    int a = year / 1000 % 10;
    int b = year / 100 % 10;
    int c = year / 10 % 10;
    int d = year % 10;
    if (a != b && a != c && a != d && b != c && b != d && c != d) {
      cout << year << endl;
      break;
    }
  }
}
