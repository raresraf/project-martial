#include <iostream>
using namespace std;
int main() {
  int number_of_stone, k = 0;
  string str;
  cin >> number_of_stone >> str;
  for (int i = 0; i < number_of_stone; i++) {
    if (str[i] == str[i + 1]) {
      k++;
    } else {
      continue;
    }
  }
  cout << k;
}
