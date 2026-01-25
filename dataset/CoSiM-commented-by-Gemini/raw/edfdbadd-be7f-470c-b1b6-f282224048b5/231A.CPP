#include <cmath>
#include <iostream>
using namespace std;
int main() {
  int num1[1000];
  int num2[1000];
  int num3[1000];
  int counter = 0;
  int y;
  int i;

  cin >> y;

  for (i = 0; i < y; i++) {
    cin >> num1[i] >> num2[i] >> num3[i];

    if (num1[i] == num2[i] && num1[i] == num3[i] && num1[i] == 1)
      counter++;
    else if (num1[i] == num2[i] && num1[i] == 1)
      counter++;
    else if (num2[i] == num3[i] && num2[i] == 1)
      counter++;
    else if (num1[i] == num3[i] && num1[i] == 1)
      counter++;
  }
  cout << counter << endl;
  return 0;
}
