#include <iostream>

using namespace std;

int main() {
  int length;
  int noluckey = 0;
  string holder;
  cin >> holder;
  length = holder.length();
  char almostlucky[length];
  for (int i = 0; i < length; i++) {
    almostlucky[i] = holder.at(i);
    if (holder.at(i) == '4' || holder.at(i) == '7') {
      noluckey++;
    }
  }
  if (noluckey == 4 || noluckey == 7) {
    cout << "YES";
  } else {
    cout << "NO";
  }
  return 0;
}