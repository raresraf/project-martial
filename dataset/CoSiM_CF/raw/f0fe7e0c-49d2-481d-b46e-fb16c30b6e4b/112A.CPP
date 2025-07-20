#include <iostream>
#include <string>

using namespace std;

int main() {
  string first;
  string second;
  int diff = 0;
  cin >> first;
  cin >> second;
  for (int i = 0; i < first.length(); i++) {
    char temp = first.at(i);
    if (temp >= 'a' && temp <= 'z') {
      first[i] = temp - 32;
    }
    temp = second.at(i);
    if (temp >= 'a' && temp <= 'z') {
      second[i] = temp - 32;
    }
    if (first[i] > second[i]) {
      diff = 1;
      break;
    } else if (first[i] < second[i]) {
      diff = -1;
      break;
    }
  }
  cout << diff;
  return 0;
}
