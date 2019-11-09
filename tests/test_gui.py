import os
import tempfile
import sys
import unittest

from qtpy.QtWidgets import QApplication
from qtpy.QtTest import QTest
from qtpy.QtCore import Qt

import quest


app = QApplication(sys.argv)


class Tests(unittest.TestCase):

    def test_run_simulation_1(self):
        form = quest.lib.tools.TransientDecayGenerator()

        pdb_filename = './data/5fwl.pdb'
        print("PDB: %s" % pdb_filename)
        form.onLoadPDB(
            pdb_filename=pdb_filename
        )

        chain_id = "K"
        chain_selector_combobox = form.pdb_selector.comboBox
        chain_id_idx = chain_selector_combobox.findText(chain_id)
        chain_selector_combobox.setCurrentIndex(chain_id_idx)

        residue_id = "135"
        residue_selector_combobox = form.pdb_selector.comboBox_2
        residue_id_idx = residue_selector_combobox.findText(residue_id)
        residue_selector_combobox.setCurrentIndex(residue_id_idx)

        atom_name = "CB"
        atom_name_selector_combobox = form.pdb_selector.comboBox_3
        atom_name_idx = atom_name_selector_combobox.findText(atom_name)
        atom_name_selector_combobox.setCurrentIndex(atom_name_idx)

        update_simulation_button = form.pushButton_3
        QTest.mouseClick(update_simulation_button, Qt.LeftButton)

    def test_run_simulation_2(self):
        form = quest.lib.tools.TransientDecayGenerator()
        pdb_filename = './data/1dg3.pdb'
        print("PDB: %s" % pdb_filename)
        form.onLoadPDB(
            pdb_filename=pdb_filename
        )

        chain_id = "A"
        chain_selector_combobox = form.pdb_selector.comboBox
        chain_id_idx = chain_selector_combobox.findText(chain_id)
        chain_selector_combobox.setCurrentIndex(chain_id_idx)

        residue_id = "344"
        residue_selector_combobox = form.pdb_selector.comboBox_2
        residue_id_idx = residue_selector_combobox.findText(residue_id)
        residue_selector_combobox.setCurrentIndex(residue_id_idx)

        atom_name = "CB"
        atom_name_selector_combobox = form.pdb_selector.comboBox_3
        atom_name_idx = atom_name_selector_combobox.findText(atom_name)
        atom_name_selector_combobox.setCurrentIndex(atom_name_idx)

        update_simulation_button = form.pushButton_3
        QTest.mouseClick(update_simulation_button, Qt.LeftButton)

    """

    def test_calculation_1(self):

        self.assertAlmostEqual(self.form.doubleSpinBox_10.value(), 0.7545, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_9.value(), 0.1907, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_6.value(), 1.0164, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_8.value(), 0.0424, places=2)

    def test_calculation_2(self):
        check_box = self.form.checkBox
        check_box.setCheckState(True)

        okWidget = self.form.pushButton
        QTest.mouseClick(okWidget, Qt.LeftButton)

        self.assertAlmostEqual(self.form.doubleSpinBox_10.value(), 0.6605, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_9.value(), 0.1398, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_6.value(), 0.9951, places=2)
        self.assertAlmostEqual(self.form.doubleSpinBox_8.value(), 0.0363, places=2)

    def setFormToZero(self):
        '''Set all ingredients to zero in preparation for setting just one
        to a nonzero value.
        '''
        self.form.ui.tequilaScrollBar.setValue(0)
        self.form.ui.tripleSecSpinBox.setValue(0)
        self.form.ui.limeJuiceLineEdit.setText("0.0")
        self.form.ui.iceHorizontalSlider.setValue(0)

    def test_tequilaScrollBar(self):
        '''Test the tequila scroll bar'''
        self.setFormToZero()

        # Test the maximum.  This one goes to 11.
        self.form.ui.tequilaScrollBar.setValue(12)
        self.assertEqual(self.form.ui.tequilaScrollBar.value(), 11)

        # Test the minimum of zero.
        self.form.ui.tequilaScrollBar.setValue(-1)
        self.assertEqual(self.form.ui.tequilaScrollBar.value(), 0)

        self.form.ui.tequilaScrollBar.setValue(5)

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 5)

    def test_tripleSecSpinBox(self):
        '''Test the triple sec spin box.
        Testing the minimum and maximum is left as an exercise for the reader.
        '''
        self.setFormToZero()
        self.form.ui.tripleSecSpinBox.setValue(2)

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 2)

    def test_limeJuiceLineEdit(self):
        '''Test the lime juice line edit.
        Testing the minimum and maximum is left as an exercise for the reader.
        '''
        self.setFormToZero()
        # Clear and then type "3.5" into the lineEdit widget
        self.form.ui.limeJuiceLineEdit.clear()
        QTest.keyClicks(self.form.ui.limeJuiceLineEdit, "3.5")

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 3.5)

    def test_iceHorizontalSlider(self):
        '''Test the ice slider.
        Testing the minimum and maximum is left as an exercise for the reader.
        '''
        self.setFormToZero()
        self.form.ui.iceHorizontalSlider.setValue(4)

        # Push OK with the left mouse button
        okWidget = self.form.ui.buttonBox.button(self.form.ui.buttonBox.Ok)
        QTest.mouseClick(okWidget, Qt.LeftButton)
        self.assertEqual(self.form.jiggers, 4)

    def test_liters(self):
        '''Test the jiggers-to-liters conversion.'''
        self.setFormToZero()
        self.assertAlmostEqual(self.form.liters, 0.0)
        self.form.ui.iceHorizontalSlider.setValue(1)

    def test_blenderSpeedButtons(self):
        '''Test the blender speed buttons'''
        self.form.ui.speedButton1.click()
        self.assertEqual(self.form.speedName, "&Mix")
        self.form.ui.speedButton2.click()
    """


if __name__ == "__main__":
    unittest.main()
