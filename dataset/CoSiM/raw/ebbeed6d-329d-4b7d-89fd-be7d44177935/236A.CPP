#include <iostream>

using namespace std;

int main() {
  string name;
  int numberOfDistinct = 0;
  cin >> name;
  int l = name.length();
  for (int i = 0; i < l; i++) {
    bool isDistinct = true;
    for (int j = i; j < l; j++) {
      if (j != i && name[j] == name[i]) {
        isDistinct = false;
      }
    }
    if (isDistinct) {
      numberOfDistinct += 1;
    }
  }
  cout << (numberOfDistinct % 2 == 0 ? "CHAT WITH HER!" : "IGNORE HIM!")
       << endl;
}