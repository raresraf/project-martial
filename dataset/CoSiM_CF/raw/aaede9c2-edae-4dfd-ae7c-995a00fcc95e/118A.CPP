#include <algorithm>
#include <iostream>
#include <string>

using namespace std;

int main() {
  string palabra, novocals = "", dotword = "";
  cin >> palabra;
  transform(palabra.begin(), palabra.end(), palabra.begin(), ::tolower);
  for (int i = 0; i < palabra.size(); i++) {
    switch (palabra[i]) {
    case 'a':
      continue;
    case 'o':
      continue;
    case 'y':
      continue;
    case 'e':
      continue;
    case 'u':
      continue;
    case 'i':
      continue;
    default:
      novocals = novocals + palabra[i];
    }
  }
  for (int i = 0; i < novocals.size(); i++) {
    dotword = dotword + "." + novocals[i];
  }
  cout << dotword;

  return 0;
}
