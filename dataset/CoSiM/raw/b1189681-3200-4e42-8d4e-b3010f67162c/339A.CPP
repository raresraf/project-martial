#include <iostream>
#include <string>
using namespace std;

int main() {
  string word, w1 = "", w2 = "", w3 = "";
  cin >> word;
  int size = word.size() / 2 + 1, temp = 0;

  while (size--) {
    if (word[temp] == '1')
      w1 += "1+";
    else if (word[temp] == '2')
      w2 += "2+";
    else
      w3 += "3+";

    temp += 2;
  }
  if (w2.size() > 0 || w3.size() > 0) {
    cout << w1;
    if (w3.size() == 0)
      cout << w2.substr(0, w2.size() - 1);
    else
      cout << w2;
  } else
    cout << w1.substr(0, w1.size() - 1);

  cout << w3.substr(0, w3.size() - 1) << endl;
  return 0;
}
