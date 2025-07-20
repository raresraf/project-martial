#include <iostream>
#include <string>

using namespace std;

int main() {
  string st;

  cin >> st;

  for (int i = 0; i < st.size(); i++) {
    char sub = static_cast<char>(tolower(st[i]));

    if (!(sub == 'a' || sub == 'e' || sub == 'i' || sub == 'o' || sub == 'u' ||
          sub == 'y')) {
      cout << "." << sub;
    }
  }

  cout << endl;
}
