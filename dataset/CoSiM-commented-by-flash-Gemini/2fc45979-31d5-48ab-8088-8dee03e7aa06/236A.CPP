#include <iostream>
#include <string>
using namespace std;
int main() {
  string x;
  cin >> x;

  for (int i = 0; i < x.size(); i++) {
    for (int j = i + 1; j < x.size(); j++) {
      if (x[i] == x[j]) {
        x.erase(j, 1);
        j--;
      }
    }
  }
  if (x.size() % 2 == 0) {
    cout << "CHAT WITH HER!";
  } else
    cout << "IGNORE HIM!";
  return 0;
}
