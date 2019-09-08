echo "Getting data..."

# Create data directories
echo "Creating default directories..."
mkdir ../../data
mkdir ../../data/images
mkdir ../../data/ground_truths
mkdir ../../data/ground_truths/MA
mkdir ../../data/ground_truths/HE
mkdir ../../data/ground_truths/EX
mkdir ../../data/ground_truths/SE

# Copy necessary data from downloaded data
echo "Copying necessary data in default directories..."
cp ../../A.\ Segmentation/1.\ Original\ Images/a.\ Training\ Set/* ../../data/images/
cp ../../A.\ Segmentation/2.\ All\ Segmentation\ Groundtruths/a.\ Training\ Set/1.\ Microaneurysms/* ../../data/ground_truths/MA/
cp ../../A.\ Segmentation/2.\ All\ Segmentation\ Groundtruths/a.\ Training\ Set/2.\ Haemorrhages/* ../../data/ground_truths/HE/
cp ../../A.\ Segmentation/2.\ All\ Segmentation\ Groundtruths/a.\ Training\ Set/3.\ Hard\ Exudates/* ../../data/ground_truths/EX/
cp ../../A.\ Segmentation/2.\ All\ Segmentation\ Groundtruths/a.\ Training\ Set/4.\ Soft\ Exudates/* ../../data/ground_truths/SE/

# Remove download data (not needed anymore)
echo "Removing downloaded data which are no needed anymore..."
rm -rf ../../A.\ Segmentation

# Quick check verifying that the data have been correctly copied
python check_data.py

echo "Done!"