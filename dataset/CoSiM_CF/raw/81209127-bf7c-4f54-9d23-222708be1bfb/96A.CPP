#include <cstring>
#include <iostream>
using namespace std;

int main() {
  string teams;
  cin >> teams;

  int count = 1;
  int length = teams.length();

  for (int i = 0; i < length - 1; i++) {
    if (teams[i] != teams[i + 1]) {
      count = 1;
    } else {
      count++;
    }
    if (count == 7) {
      cout << "YES";
      return 0;
    }
  }
  cout << "NO";
  return 0;
}