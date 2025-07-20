#include <iostream>

using namespace std;

int main() {
  string number;
  int lucky_numbers = 0;

  cin >> number;

  for (int i = 0; i < number.size(); i++)
    if (number[i] == '7' || number[i] == '4')
      lucky_numbers++;

  if (lucky_numbers == 7 || lucky_numbers == 4)
    cout << "YES";
  else
    cout << "NO";

  return 0;
}