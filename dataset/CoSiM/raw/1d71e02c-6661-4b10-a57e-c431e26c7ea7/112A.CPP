#include <iostream>
#include <string>
using namespace std;
int main() {

  string first, second;
  cin >> first >> second;

  int flag = 0;

  for (int i = 0; i < first.length(); i++) {
    if (first[i] >= 'a' && first[i] <= 'z') {
      first[i] -= 'a';
    } else if (first[i] >= 'A' && first[i] <= 'Z') {
      first[i] -= 'A';
    }
    if (second[i] >= 'a' && second[i] <= 'z') {
      second[i] -= 'a';
    } else if (second[i] >= 'A' && second[i] <= 'Z') {
      second[i] -= 'A';
    }
    if ((int)first[i] > (int)second[i]) {
      flag = 1;
      break;
    } else if ((int)first[i] < second[i]) {
      flag = -1;
      break;
    }
  }

  cout << flag << endl;
}
